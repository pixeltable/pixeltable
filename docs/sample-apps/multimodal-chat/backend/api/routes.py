import atexit
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from settings import ALLOWED_TYPES, logger

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.iterators import DocumentSplitter
from pixeltable.iterators.string import StringSplitter

router = APIRouter()

# Configure upload directory
TEMP_DIR = tempfile.mkdtemp()
atexit.register(shutil.rmtree, TEMP_DIR)
logger.info(f"Temporary directory: {Path(TEMP_DIR).absolute()}")


class ChatMessage(BaseModel):
    message: str


# Initialize Pixeltable
pxt.drop_dir("chatbot", force=True)
pxt.create_dir("chatbot")
logger.info("Created Pixeltable directory")

docs_table = pxt.create_table(
    "chatbot.documents",
    {
        "document": pxt.Document,
        "video": pxt.Video,
        "audio": pxt.Audio,
        "question": pxt.String,
    },
)

conversations = pxt.create_table(
    "chatbot.conversations",
    {"role": pxt.String, "content": pxt.String, "timestamp": pxt.Timestamp},
)


# Create prompt function
@pxt.udf
def create_prompt(
    doc_context: list[dict],
    video_context: list[dict],
    audio_context: list[dict],
    question: str,
) -> str:
    context_parts = []

    if doc_context:
        context_parts.append(
            "Document Context:\n"
            + "\n\n".join(
                item["text"] for item in doc_context if item and "text" in item
            )
        )

    if video_context:
        context_parts.append(
            "Video Context:\n"
            + "\n\n".join(
                item["text"] for item in video_context if item and "text" in item
            )
        )

    if audio_context:
        context_parts.append(
            "Audio Context:\n"
            + "\n\n".join(
                item["text"] for item in audio_context if item and "text" in item
            )
        )

    full_context = (
        "\n\n---\n\n".join(context_parts)
        if context_parts
        else "No relevant context found."
    )

    return f"Context:\n{full_context}\n\nQuestion:\n{question}"


# Add computed columns for video processing
docs_table.add_computed_column(audio_extract=extract_audio(docs_table.video, format="mp3"))
docs_table.add_computed_column(transcription=openai.transcriptions(
    audio=docs_table.audio_extract, model="whisper-1"
))
docs_table.add_computed_column(audio_transcription=openai.transcriptions(
    audio=docs_table.audio, model="whisper-1"
))
docs_table.add_computed_column(audio_transcription_text=docs_table.audio_transcription.text)
docs_table.add_computed_column(transcription_text=docs_table.transcription.text)

logger.info("Created documents table")

# Create chunks view
chunks_view = pxt.create_view(
    "chatbot.chunks",
    docs_table,
    iterator=DocumentSplitter.create(
        document=docs_table.document,
        separators="sentence",
        metadata="title,heading,sourceline",
    ),
)

logger.info("Created chunks view")


# Create view for chunking transcriptions
transcription_chunks = pxt.create_view(
    "chatbot.transcription_chunks",
    docs_table,
    iterator=StringSplitter.create(
        text=docs_table.transcription_text, separators="sentence"
    ),
)

audio_chunks = pxt.create_view(
    "chatbot.audio_chunks",
    docs_table,
    iterator=StringSplitter.create(
        text=docs_table.audio_transcription_text, separators="sentence"
    ),
)

# Add embedding index to document chunks
chunks_view.add_embedding_index(
    "text", string_embed=sentence_transformer.using(model_id="intfloat/e5-large-v2")
)

# Add embedding index to transcription chunks
transcription_chunks.add_embedding_index(
    "text", string_embed=sentence_transformer.using(model_id="intfloat/e5-large-v2")
)

audio_chunks.add_embedding_index(
    "text", string_embed=sentence_transformer.using(model_id="intfloat/e5-large-v2")
)

logger.info("Added embedding index")


@pxt.query
def get_chat_history():
    return conversations.order_by(
        conversations.timestamp
    ).select(
        role=conversations.role,
        content=conversations.content
    )

@pxt.udf
def create_messages(history: list[dict], prompt: str) -> list[dict]:
    messages = [{
        'role': 'system',
        'content': '''You are a helpful AI assistant maintaining conversation context while answering questions based on provided sources.'''
    }]

    # Add historical messages
    messages.extend({
        'role': msg['role'],
        'content': msg['content']
    } for msg in history)

    # Add current prompt
    messages.append({
        'role': 'user',
        'content': prompt
    })

    return messages

# Setup similarity search query
@pxt.query
def get_relevant_chunks(query_text: str):
    sim = chunks_view.text.similarity(query_text)
    return (
        chunks_view.order_by(sim, asc=False).select(chunks_view.text, sim=sim).limit(20)
    )


@pxt.query
def get_relevant_transcript_chunks(query_text: str):
    sim = transcription_chunks.text.similarity(query_text)
    return (
        transcription_chunks.order_by(sim, asc=False)
        .select(transcription_chunks.text, sim=sim)
        .limit(20)
    )


@pxt.query
def get_relevant_audio_chunks(query_text: str):
    sim = audio_chunks.text.similarity(query_text)
    return (
        audio_chunks.order_by(sim, asc=False)
        .select(audio_chunks.text, sim=sim)
        .limit(20)
    )


# Add computed columns
docs_table.add_computed_column(context_doc=get_relevant_chunks(docs_table.question))
docs_table.add_computed_column(context_video=get_relevant_transcript_chunks(docs_table.question))
docs_table.add_computed_column(context_audio=get_relevant_audio_chunks(docs_table.question))
docs_table.add_computed_column(prompt=create_prompt(
    docs_table.context_doc,
    docs_table.context_video,
    docs_table.context_audio,
    docs_table.question,
))
docs_table.add_computed_column(chat_history=get_chat_history())
docs_table.add_computed_column(messages=create_messages(
    docs_table.chat_history,
    docs_table.prompt
))
docs_table.add_computed_column(response=openai.chat_completions(
    messages=docs_table.messages,
    model="gpt-4o-mini",
))

# Extract the answer text from the API response
docs_table.add_computed_column(answer=docs_table.response.choices[0].message.content)


logger.info("Setup complete")


@router.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process document files with Pixeltable's native support."""
    logger.info(f"Received file upload request: {file.filename}, type: {file.content_type}")

    if not file.content_type in ALLOWED_TYPES['document']:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document format. Supported formats are: PDF, MD, HTML, TXT, XML"
        )

    try:
        # Save file to temp directory
        file_path = Path(TEMP_DIR) / file.filename
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process document
        abs_path = str(file_path.absolute()).replace(os.sep, '/')
        logger.info(f"Processing document: {abs_path}")

        docs_table = pxt.get_table('chatbot.documents')
        docs_table.insert([{'document': abs_path}])

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded and processed {file.filename}",
                "filename": file.filename,
                "type": "document"
            }
        )

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Error processing file: {str(e)}")


@router.get("/api/files")
async def list_files():
    try:
        docs_table = pxt.get_table("chatbot.documents")
        doc_results = docs_table.select(docs_table.document).collect().to_pandas()
        video_results = docs_table.select(docs_table.video).collect().to_pandas()

        files = []

        # Process documents
        for _, row in doc_results.iterrows():
            path = row["document"]
            if not path:
                continue

            try:
                file_path = Path(path)
                if file_path.exists():
                    files.append(
                        {
                            "id": str(hash(file_path.name)),
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "type": "document",
                            "status": "success",
                            "uploadProgress": 100,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing document path {path}: {e}")
                continue

        # Process videos
        for _, row in video_results.iterrows():
            path = row["video"]
            if not path:
                continue

            try:
                file_path = Path(path)
                if file_path.exists():
                    files.append(
                        {
                            "id": str(hash(file_path.name)),
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "type": "video",
                            "status": "success",
                            "uploadProgress": 100,
                        }
                    )
            except Exception as e:
                logger.error(f"Error processing video path {path}: {e}")
                continue

        # Sort files by type and name
        files.sort(key=lambda x: (x["type"], x["name"]))

        return JSONResponse(status_code=200, content={"files": files})

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(500, f"Error listing files: {str(e)}")


@router.post("/api/videos/upload")
async def upload_video(file: UploadFile = File(...)):
    if not any(file.content_type.startswith(vtype) for vtype in ALLOWED_TYPES["video"]):
        raise HTTPException(400, "Invalid video format")

    try:
        # Save video file
        file_path = Path(TEMP_DIR) / file.filename
        logger.info(f"Saving video to {file_path}")

        # Read and save file
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Save to Pixeltable
        docs_table = pxt.get_table("chatbot.documents")
        full_path = str(file_path.absolute())

        # Insert into Pixeltable with normalized path
        docs_table.insert([{"video": full_path.replace(os.sep, "/")}])

        logger.info(f"Video saved and inserted into Pixeltable: {full_path}")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded video: {file.filename}",
                "filename": file.filename,
                "path": str(file_path),
            },
        )

    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Error uploading video: {str(e)}")


@router.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not any(file.content_type.startswith(atype) for atype in ALLOWED_TYPES["audio"]):
        raise HTTPException(400, "Invalid audio format")

    try:
        file_path = Path(TEMP_DIR) / file.filename
        normalized_path = str(file_path.absolute()).replace(os.sep, "/")

        # Save file
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Insert into Pixeltable
        docs_table = pxt.get_table("chatbot.documents")
        docs_table.insert([{"audio": normalized_path}])

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully uploaded audio: {file.filename}",
                "filename": file.filename,
                "path": normalized_path,
            },
        )

    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Error uploading audio: {str(e)}")


async def get_answer(question: str) -> str:
    docs_table = pxt.get_table("chatbot.documents")

    try:
        # Insert question
        docs_table.insert([{"question": question}])

        # Get answer using Pixeltable's collect() method
        result = (
            docs_table.select(docs_table.answer)
            .where(docs_table.question == question)
            .collect()
        )

        # Check if there are any results
        if len(result) == 0:
            return "No response was generated. Please try asking another question."

        # Get the first answer from the results
        answer = result["answer"][0]

        # Validate the answer
        if not answer or answer.strip() == "":
            return (
                "An empty response was generated. Please try asking another question."
            )

        return answer

    except Exception as e:
        logger.error(f"Error getting answer: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )


@router.post("/api/chat")
async def chat(message: ChatMessage):
    try:
        # Store user message
        conversations.insert(
            [{"role": "user", "content": message.message, "timestamp": datetime.now()}]
        )

        # Get response from Pixeltable
        response = await get_answer(message.message)

        # Store assistant response
        conversations.insert(
            [{"role": "assistant", "content": response, "timestamp": datetime.now()}]
        )

        return JSONResponse(
            status_code=200, content={"response": response, "used_files": []}
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(500, str(e))

@router.get("/")
async def root():
    logger.info("Root endpoint accessed")
    response = "Pixeltable Multimodal API"
    return response

@router.get("/health")
async def health_check():
    return {"status": "ok"}
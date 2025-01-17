# pip install boto3 pixeltable tiktoken openai openai-whisper spacy sentence-transformers
# Install the spacy model before running: python -m spacy download en_core_web_sm

import pixeltable as pxt
from pixeltable.functions import openai
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.iterators.string import StringSplitter

import spacy

nlp = spacy.load("en_core_web_sm")

# Initialize Pixeltable
pxt.drop_dir("agent", force=True)
pxt.create_dir("agent")

############################################################
# Create Audio Knowledge Base
############################################################

# Create audio table
audio_t = pxt.create_table("agent.audio", {"audio_file": pxt.Audio})

# Audio-to-text
audio_t.add_computed_column(
    transcription=whisper.transcribe(audio=audio_t.audio_file, model="base.en")
)

# Split sentences
sentences_view = pxt.create_view(
    "agent.audio_sentence_chunks",
    audio_t,
    iterator=StringSplitter.create(
        text=audio_t.transcription.text, separators="sentence"
    ),
)

# Create embeddings
embed_model = sentence_transformer.using(model_id="intfloat/e5-large-v2")

# Viola, a self-updating knowledge base
sentences_view.add_embedding_index(column="text", string_embed=embed_model)

############################################################
# Agentic RAG (Query-as-a-Tool)
############################################################


# Retrieval on the audio transcription
@pxt.query
def top_k(query_text: str):
    sim = sentences_view.text.similarity(query_text)
    return (
        sentences_view.order_by(sim, asc=False)
        .select(sentences_view.text, sim=sim)
        .limit(10)
    )


# Create Tools
tools = pxt.tools(top_k)

############################################################
# Create Agent
############################################################

# Create agent table
conversations = pxt.create_table(
    path_str="agent.conversations",
    schema_or_df={"prompt": pxt.String},
    if_exists="ignore",
)

# Initial tool calling to retrieve documents
messages = [{"role": "user", "content": conversations.prompt}]

conversations.add_computed_column(
    tool_response=openai.chat_completions(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice=tools.choice(required=True),
    )
)

conversations.add_computed_column(
    retrieved_passages=openai.invoke_tools(tools, conversations.tool_response)
)


# Helper function to create final prompt
@pxt.udf
def create_prompt(question: str, passages: list[dict]) -> str:
    return f"""
    QUESTION:

    {question}

    RELEVANT PASSAGES:

    {passages}
    """


# Add final prompt to agent table
conversations.add_computed_column(
    final_prompt=create_prompt(conversations.prompt, conversations.retrieved_passages)
)

# Final response generation
final_messages = [
    {
        "role": "system",
        "content": "Answer the user's question based on the retrieved passages.",
    },
    {"role": "user", "content": conversations.final_prompt},
]

conversations.add_computed_column(
    final_response=openai.chat_completions(model="gpt-4o-mini", messages=final_messages)
)

conversations.add_computed_column(
    answer=conversations.final_response.choices[0].message.content
)

############################################################
# Chat Interface
############################################################


def chat(message: str) -> str:
    conversations.insert([{"prompt": message}])
    print(conversations.retrieved_passages.show())
    print(conversations.answer.show())


if __name__ == "__main__":

    # Audio ingestion pipeline
    audio_t.insert(
        [
            {
                "audio_file": "s3://pixeltable-public/audio/10-minute tour of Pixeltable.mp3"
            }
        ]
    )

    print(chat("Summarize the tour"))

# Standard library imports
import os
import io
import base64
import tempfile
import logging
from pathlib import Path
from typing import Optional, Literal

# Third-party library imports
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import PIL.Image

# Local imports
import pixeltable as pxt
from pixeltable.iterators import FrameIterator
from pixeltable.functions.huggingface import clip_image, clip_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Create temp directory
TEMP_DIR = tempfile.mkdtemp()


# Embedding functions
@pxt.expr_udf
def embed_image(img: PIL.Image.Image):
    return clip_image(img, model_id='openai/clip-vit-base-patch32')


@pxt.expr_udf
def str_embed(s: str):
    return clip_text(s, model_id='openai/clip-vit-base-patch32')


# Initialize Pixeltable
pxt.drop_dir('video_search', force=True)
pxt.create_dir('video_search')

# Create video table
video_table = pxt.create_table('video_search.videos', {'video': pxt.Video})

# Create frames view
frames_view = pxt.create_view(
    'video_search.frames', video_table, iterator=FrameIterator.create(video=video_table.video, fps=1)
)

# Add embedding index
frames_view.add_embedding_index('frame', string_embed=str_embed, image_embed=embed_image)


@app.post('/api/process-video')
async def process_video(file: UploadFile = File(...)):
    try:
        video_table = pxt.get_table('video_search.videos')

        # Save video file
        temp_path = Path(TEMP_DIR) / file.filename
        with temp_path.open('wb') as buffer:
            content = await file.read()
            buffer.write(content)

        # Insert video
        video_table.insert([{'video': str(temp_path)}])

        return {'message': 'Video processed successfully'}

    except Exception as e:
        logger.error(f'Error processing video: {e}')
        raise HTTPException(status_code=500, detail=str(e))


class SearchQuery(BaseModel):
    query: str
    search_type: Literal['text', 'image']
    num_results: int


@app.post('/api/search')
async def search_video(
    query: Optional[UploadFile] = File(None),
    search_type: Literal['text', 'image'] = Form(...),
    num_results: int = Form(...),
):
    try:
        frames_view = pxt.get_table('video_search.frames')

        if search_type == 'text':
            # Handle text search
            text_query = query.filename if query else ''  # The text query is sent in the filename field
            sim = frames_view.frame.similarity(text_query)
        else:
            # Handle image search
            if not query:
                raise HTTPException(status_code=400, detail='No image provided')

            # Read the image file
            contents = await query.read()
            image = PIL.Image.open(io.BytesIO(contents))

            # Perform image similarity search
            sim = frames_view.frame.similarity(image)

        # Get results
        results = frames_view.order_by(sim, asc=False).limit(num_results).select(frames_view.frame).collect()

        # Convert frames to base64 for sending to frontend
        frames = []
        for row in results:
            frame = row['frame']
            if isinstance(frame, PIL.Image.Image):
                # Convert PIL Image to base64
                buffered = io.BytesIO()
                frame.save(buffered, format='PNG')
                img_str = base64.b64encode(buffered.getvalue()).decode()
                frames.append(f'data:image/png;base64,{img_str}')

        return {'frames': frames, 'success': True}

    except Exception as e:
        logger.error(f'Search error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)

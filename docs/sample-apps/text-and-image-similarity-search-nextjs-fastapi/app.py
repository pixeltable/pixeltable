# Standard library imports
import io
import json
import tempfile
from pathlib import Path
from typing import Literal, Optional

# Image handling library
import PIL.Image

# Third-party library imports
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# --- Pixeltable Imports ---
# The core pixeltable library
import pixeltable as pxt

# Specific functions: HuggingFace integration (for CLIP model)
from pixeltable.functions.huggingface import clip

# Iterators: Tools to transform data, like extracting frames from video
from pixeltable.iterators import FrameIterator

# --- End Pixeltable Imports ---

# Initialize FastAPI app (Standard FastAPI)
app = FastAPI()

# Configure CORS (Standard FastAPI - Allows frontend at localhost:3000 to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Create a temporary directory for storing uploaded files before Pixeltable processes them
TEMP_DIR = tempfile.mkdtemp()

# --- Pixeltable Setup ---
# Pixeltable stores its metadata and generated data in directories.
# We ensure a clean state by removing any previous directory.
pxt.drop_dir('media_search', force=True)
# Create the main directory for our Pixeltable data.
pxt.create_dir('media_search')

# --- Define Pixeltable Tables and Views ---
# Define a Pixeltable 'table' to store uploaded videos.
# Tables have a schema, similar to SQL tables, but with support for media types.
video_table = pxt.create_table(
    'media_search.videos',
    {
        # 'video' column stores video data; Pixeltable handles file references/storage.
        'video': pxt.Video
    },
)

# Define a Pixeltable 'view'. Views are derived from tables or other views.
# Here, we use FrameIterator to automatically extract frames from videos in 'video_table'.
# The view effectively creates a new "table" where each row represents one frame.
frames_view = pxt.create_view(
    'media_search.frames',  # Name of the view
    video_table,  # Source table
    # The iterator defines how to generate rows for the view.
    # FrameIterator extracts frames at a specified FPS.
    iterator=FrameIterator.create(video=video_table.video, fps=1),
)

# Create an embedding index on the 'frame' column of the 'frames_view'.
# This is the core of the similarity search functionality.
# Pixeltable will automatically:
# 1. Run the specified CLIP model on each frame.
# 2. Store the resulting embedding vector.
# 3. Use an efficient vector index for fast similarity lookups.
# The CLIP model allows *cross-modal* search (text-to-image and image-to-image).
frames_view.add_embedding_index(
    'frame',  # The column containing the image data (video frames)
    # Specify the embedding function. '.using()' configures the function.
    embedding=clip.using(model_id='openai/clip-vit-base-patch32'),  # Use OpenAI's CLIP model
)

# Define a Pixeltable table for storing uploaded images (e-commerce/tagging example).
image_table = pxt.create_table('media_search.images', {'image': pxt.Image, 'tags': pxt.Json})

# Add a similar embedding index to the image table.
# This allows searching uploaded images using text or other images.
image_table.add_embedding_index(
    'image',  # The column with image data
    embedding=clip.using(model_id='openai/clip-vit-base-patch32'),  # Use the same CLIP model
)
# --- End Pixeltable Setup ---


# --- API Endpoints ---


# Endpoint to upload and process a video file.
@app.post('/api/process-video')
async def process_video(file: UploadFile = File(...)):  # noqa: B008
    try:
        # Get a reference to the Pixeltable table.
        video_table = pxt.get_table('media_search.videos')

        # Standard FastAPI file handling: save the uploaded video temporarily.
        temp_path = Path(TEMP_DIR) / file.filename
        with temp_path.open('wb') as buffer:
            content = await file.read()
            buffer.write(content)

        # Insert the video file path into the Pixeltable table.
        # Pixeltable handles copying/referencing the file.
        # Any views derived from this table (like 'frames_view') will
        # automatically update and process the new video data incrementally.
        video_table.insert([{'video': str(temp_path)}])

        # The frame extraction and embedding generation happen automatically
        # in the background due to the view and index definitions.
        return {'message': 'Video processed successfully and frames are being indexed.'}

    except Exception as e:
        print(f'Error processing video: {e}')
        raise HTTPException(status_code=500, detail=str(e)) from e


# Endpoint to upload an image with tags.
@app.post('/api/upload-image')
async def upload_image(tags: str = Form(...), file: UploadFile = File(...)):  # noqa: B008
    try:
        # Get a reference to the image table.
        image_table = pxt.get_table('media_search.images')

        # Parse the incoming JSON string of tags into a Python list.
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list) or not all(isinstance(tag, str) for tag in tags_list):
                raise ValueError('Tags must be a JSON array of strings.')
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f'Invalid tags format: {e}') from e

        # Save the uploaded image temporarily.
        temp_path = Path(TEMP_DIR) / file.filename
        with temp_path.open('wb') as buffer:
            content = await file.read()
            buffer.write(content)

        # Insert the image path and tags list.
        image_table.insert([{'image': str(temp_path), 'tags': tags_list}])

        return {'message': 'Image uploaded and tagged successfully.'}

    except Exception as e:
        print(f'Error uploading image: {e}')
        raise HTTPException(status_code=500, detail=str(e)) from e


# Endpoint to search video frames.
@app.post('/api/search')
async def search_video(
    query: Optional[UploadFile] = File(None),  # noqa: B008 # Can be text (in filename) or image file
    search_type: Literal['text', 'image'] = Form(...),
    num_results: int = Form(...),
):
    try:
        # Get a reference to the 'frames' view.
        frames_view = pxt.get_table('media_search.frames')  # Use get_table for views too

        # Determine the query input (text string or PIL image)
        if search_type == 'text':
            text_query = query.filename if query else ''
            if not text_query:
                raise HTTPException(status_code=400, detail='No text query provided')
            query_input = text_query
        else:  # Image search
            if not query:
                raise HTTPException(status_code=400, detail='No image provided')
            contents = await query.read()
            query_input = PIL.Image.open(io.BytesIO(contents))  # Use PIL image object

        # Calculate similarity using the embedding index on the 'frame' column.
        # Pixeltable's '.similarity()' method takes either text or an image
        # and efficiently finds the most similar items using the pre-built index.
        sim = frames_view.frame.similarity(query_input)

        # Build the Pixeltable query:
        # 1. Order results by the calculated similarity (descending).
        # 2. Limit the number of results.
        # 3. Select the columns needed. Here, we encode the frame directly.
        results = (
            frames_view.order_by(sim, asc=False)
            .limit(num_results)
            # Use the built-in '.b64_encode()' method on the image column
            # for efficient encoding within the query itself.
            .select(encoded_frame=frames_view.frame.b64_encode('png'))
            # '.collect()' executes the query and retrieves the results.
            .collect()
        )

        # Format the results for the frontend.
        encoded_frames = [f'data:image/png;base64,{row["encoded_frame"]}' for row in results]

        return {'frames': encoded_frames, 'success': True}

    except Exception as e:
        print(f'Search error: {e!s}')
        raise HTTPException(status_code=500, detail=str(e)) from e


# Endpoint to search uploaded images.
@app.post('/api/search-images')
async def search_images(
    query: Optional[UploadFile] = File(None),  # noqa: B008 # Text (in filename) or image file
    search_type: Literal['text', 'image'] = Form(...),
    num_results: int = Form(...),
    similarity_threshold: float = Form(0.5),  # Add similarity threshold parameter
):
    try:
        # Get a reference to the image table.
        image_table = pxt.get_table('media_search.images')
        # Start building the query from the base table.
        base_query = image_table

        # Determine query input (text or image).
        if search_type == 'text':
            text_query = query.filename if query else ''
            if not text_query:
                raise HTTPException(status_code=400, detail='No text query provided')
            query_input = text_query
        else:  # Image search
            if not query:
                raise HTTPException(status_code=400, detail='No image provided for search')
            contents = await query.read()
            query_input = PIL.Image.open(io.BytesIO(contents))

        # Calculate similarity using the index on the 'image' column.
        sim = base_query.image.similarity(query_input)

        # Build the final query:
        # Order by similarity, limit results, and select needed columns.
        results = (
            base_query.order_by(sim, asc=False)
            .limit(num_results)
            .select(encoded_image=base_query.image.b64_encode('png'), tags=base_query.tags)
            # Execute the query.
            .where(sim > similarity_threshold)  # Use the parameter here
            .collect()
        )

        # Prepare response directly from encoded results
        output_images = [
            {'image': f'data:image/png;base64,{row["encoded_image"]}', 'tags': row['tags']} for row in results
        ]

        return {'images': output_images, 'success': True}

    except Exception as e:
        print(f'Image search error: {e!s}')
        raise HTTPException(status_code=500, detail=str(e)) from e


# Standard Python entry point for running the FastAPI app with uvicorn
if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000)

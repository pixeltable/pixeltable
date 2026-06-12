import json
import sys
from pathlib import Path
from typing import Literal

import config
import schema  # noqa: F401
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

_SAMPLE_APPS = Path(__file__).resolve().parents[2]
if str(_SAMPLE_APPS) not in sys.path:
    sys.path.insert(0, str(_SAMPLE_APPS))

import upload_utils  # noqa: E402

import pixeltable as pxt  # noqa: E402

router = APIRouter()
TEMP_DIR = upload_utils.create_upload_temp_dir()


@router.post('/api/process-video')
def process_video(file: UploadFile = File(...)):  # noqa: B008
    try:
        video_table = pxt.get_table(f'{config.APP_NAMESPACE}.videos')
        temp_path = upload_utils.save_upload(file, TEMP_DIR)
        video_table.insert([{'video': str(temp_path)}])
        return {'message': 'Video processed successfully and frames are being indexed.'}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post('/api/upload-image')
def upload_image(tags: str = Form(...), file: UploadFile = File(...)):  # noqa: B008
    try:
        image_table = pxt.get_table(f'{config.APP_NAMESPACE}.images')
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list) or not all(isinstance(tag, str) for tag in tags_list):
                raise ValueError('Tags must be a JSON array of strings.')
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f'Invalid tags format: {e}') from e

        temp_path = upload_utils.save_upload(file, TEMP_DIR)
        image_table.insert([{'image': str(temp_path), 'tags': tags_list}])
        return {'message': 'Image uploaded and tagged successfully.'}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post('/api/search')
def search_video(
    query: UploadFile | None = File(None),  # noqa: B008
    search_type: Literal['text', 'image'] = Form(...),
    num_results: int = Form(...),
):
    try:
        frames_view = pxt.get_table(f'{config.APP_NAMESPACE}.frames')

        if search_type == 'text':
            text_query = upload_utils.read_text_query(query)
            if not text_query:
                raise HTTPException(status_code=400, detail='No text query provided')
            sim = frames_view.frame.similarity(string=text_query)
        else:
            if not query:
                raise HTTPException(status_code=400, detail='No image provided')
            query_image = upload_utils.load_query_image(query)
            sim = frames_view.frame.similarity(image=query_image)

        results = (
            frames_view.order_by(sim, asc=False)
            .limit(num_results)
            .select(encoded_frame=frames_view.frame.b64_encode('png'))
            .collect()
        )
        encoded_frames = [f'data:image/png;base64,{row["encoded_frame"]}' for row in results]
        return {'frames': encoded_frames, 'success': True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post('/api/search-images')
def search_images(
    query: UploadFile | None = File(None),  # noqa: B008
    search_type: Literal['text', 'image'] = Form(...),
    num_results: int = Form(...),
    similarity_threshold: float = Form(0.5),
):
    try:
        image_table = pxt.get_table(f'{config.APP_NAMESPACE}.images')

        if search_type == 'text':
            text_query = upload_utils.read_text_query(query)
            if not text_query:
                raise HTTPException(status_code=400, detail='No text query provided')
            sim = image_table.image.similarity(string=text_query)
        else:
            if not query:
                raise HTTPException(status_code=400, detail='No image provided for search')
            query_image = upload_utils.load_query_image(query)
            sim = image_table.image.similarity(image=query_image)

        results = (
            image_table.where(sim > similarity_threshold)
            .order_by(sim, asc=False)
            .limit(num_results)
            .select(encoded_image=image_table.image.b64_encode('png'), tags=image_table.tags)
            .collect()
        )
        output_images = [
            {'image': f'data:image/png;base64,{row["encoded_image"]}', 'tags': row['tags']} for row in results
        ]
        return {'images': output_images, 'success': True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

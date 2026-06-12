import atexit
import hashlib
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from settings import ALLOWED_TYPES, logger

# Ensure backend/ is on path for config and setup_pixeltable imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
import setup_pixeltable  # noqa: F401

import pixeltable as pxt

router = APIRouter()

TEMP_DIR = Path(tempfile.mkdtemp())
atexit.register(shutil.rmtree, TEMP_DIR, ignore_errors=True)
logger.info(f'Temporary directory: {TEMP_DIR.absolute()}')

docs_table = pxt.get_table(f'{config.APP_NAMESPACE}.documents')
conversations = pxt.get_table(f'{config.APP_NAMESPACE}.conversations')


class ChatMessage(BaseModel):
    message: str


def _safe_filename(filename: str | None) -> str:
    return Path(filename or 'upload').name


def _stable_file_id(file_path: Path) -> str:
    return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]


async def _save_upload(file: UploadFile) -> Path:
    base = _safe_filename(file.filename)
    suffix = Path(base).suffix
    stem = Path(base).stem or 'upload'
    file_path = TEMP_DIR / f'{stem}_{uuid4().hex}{suffix}'
    with file_path.open('wb') as buffer:
        buffer.write(await file.read())
    return file_path


@router.post('/api/upload')
async def upload_file(file: UploadFile = File(...)):  # noqa: B008
    """Upload and process document files."""
    if file.content_type not in ALLOWED_TYPES['document']:
        raise HTTPException(status_code=400, detail='Invalid document format. Supported: PDF, MD, HTML, TXT, XML')

    file_path: Path | None = None
    try:
        file_path = await _save_upload(file)
        abs_path = str(file_path.absolute()).replace(os.sep, '/')
        docs_table.insert([{'document': abs_path}])
        return JSONResponse(
            status_code=200,
            content={
                'message': f'Successfully uploaded {file.filename}',
                'filename': file.filename,
                'type': 'document',
            },
        )
    except Exception as e:
        logger.error(f'Error processing file: {e!s}')
        if file_path is not None and file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f'Error processing file: {e!s}') from e


@router.get('/api/files')
async def list_files():
    try:
        doc_results = docs_table.select(docs_table.document).collect().to_pandas()
        video_results = docs_table.select(docs_table.video).collect().to_pandas()
        files = []

        for _, row in doc_results.iterrows():
            path = row['document']
            if not path:
                continue
            file_path = Path(path)
            if file_path.exists():
                files.append(
                    {
                        'id': _stable_file_id(file_path),
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'type': 'document',
                        'status': 'success',
                        'uploadProgress': 100,
                    }
                )

        for _, row in video_results.iterrows():
            path = row['video']
            if not path:
                continue
            file_path = Path(path)
            if file_path.exists():
                files.append(
                    {
                        'id': _stable_file_id(file_path),
                        'name': file_path.name,
                        'size': file_path.stat().st_size,
                        'type': 'video',
                        'status': 'success',
                        'uploadProgress': 100,
                    }
                )

        files.sort(key=lambda x: (x['type'], x['name']))
        return JSONResponse(status_code=200, content={'files': files})
    except Exception as e:
        logger.error(f'Error listing files: {e!s}')
        raise HTTPException(500, f'Error listing files: {e!s}') from e


@router.post('/api/videos/upload')
async def upload_video(file: UploadFile = File(...)):  # noqa: B008
    if not any(file.content_type.startswith(vtype) for vtype in ALLOWED_TYPES['video']):
        raise HTTPException(400, 'Invalid video format')

    file_path: Path | None = None
    try:
        file_path = await _save_upload(file)
        full_path = str(file_path.absolute()).replace(os.sep, '/')
        docs_table.insert([{'video': full_path}])
        return JSONResponse(
            status_code=200,
            content={
                'message': f'Successfully uploaded video: {file.filename}',
                'filename': file.filename,
                'path': full_path,
            },
        )
    except Exception as e:
        logger.error(f'Error uploading video: {e!s}')
        if file_path is not None and file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f'Error uploading video: {e!s}') from e


@router.post('/api/audio/upload')
async def upload_audio(file: UploadFile = File(...)):  # noqa: B008
    if not any(file.content_type.startswith(atype) for atype in ALLOWED_TYPES['audio']):
        raise HTTPException(400, 'Invalid audio format')

    file_path: Path | None = None
    try:
        file_path = await _save_upload(file)
        normalized_path = str(file_path.absolute()).replace(os.sep, '/')
        docs_table.insert([{'audio': normalized_path}])
        return JSONResponse(
            status_code=200,
            content={
                'message': f'Successfully uploaded audio: {file.filename}',
                'filename': file.filename,
                'path': normalized_path,
            },
        )
    except Exception as e:
        logger.error(f'Error uploading audio: {e!s}')
        if file_path is not None and file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f'Error uploading audio: {e!s}') from e


async def get_answer(question: str) -> str:
    docs_table.insert([{'question': question}])
    result = docs_table.select(docs_table.answer).where(docs_table.question == question).collect()
    if len(result) == 0:
        return 'No response was generated. Please try asking another question.'
    answer = result['answer'][0]
    if not answer or answer.strip() == '':
        return 'An empty response was generated. Please try asking another question.'
    return answer


@router.post('/api/chat')
async def chat(message: ChatMessage):
    try:
        conversations.insert([{'role': 'user', 'content': message.message, 'timestamp': datetime.now()}])
        response = await get_answer(message.message)
        conversations.insert([{'role': 'assistant', 'content': response, 'timestamp': datetime.now()}])
        return JSONResponse(status_code=200, content={'response': response, 'used_files': []})
    except Exception as e:
        logger.error(f'Error in chat: {e!s}')
        raise HTTPException(500, str(e)) from e


@router.get('/')
async def root():
    return 'Pixeltable Multimodal API'


@router.get('/health')
async def health_check():
    return {'status': 'ok'}

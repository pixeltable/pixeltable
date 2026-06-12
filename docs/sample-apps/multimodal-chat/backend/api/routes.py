import sys
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from settings import ALLOWED_TYPES, logger

# Ensure backend/ and sample-apps/ are on path for local imports
_BACKEND = Path(__file__).resolve().parent.parent
_SAMPLE_APPS = _BACKEND.parents[1]
for path in (_BACKEND, _SAMPLE_APPS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config  # noqa: E402
import setup_pixeltable  # noqa: F401, E402
import upload_utils  # noqa: E402

import pixeltable as pxt  # noqa: E402

router = APIRouter()

TEMP_DIR = upload_utils.create_upload_temp_dir()
logger.info(f'Temporary directory: {TEMP_DIR.absolute()}')

docs_table = pxt.get_table(f'{config.APP_NAMESPACE}.documents')
conversations = pxt.get_table(f'{config.APP_NAMESPACE}.conversations')


class ChatMessage(BaseModel):
    message: str


@router.post('/api/upload')
def upload_file(file: UploadFile = File(...)):  # noqa: B008
    """Upload and process document files."""
    if file.content_type not in ALLOWED_TYPES['document']:
        raise HTTPException(status_code=400, detail='Invalid document format. Supported: PDF, MD, HTML, TXT, XML')

    file_path: Path | None = None
    try:
        file_path = upload_utils.save_upload(file, TEMP_DIR)
        docs_table.insert([{'document': upload_utils.normalize_path(file_path)}])
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
def list_files():
    try:
        results = docs_table.select(docs_table.uuid, docs_table.document, docs_table.video).collect()
        files = []

        for row in results:
            file_id = str(row['uuid'])
            for col_name, file_type in [('document', 'document'), ('video', 'video')]:
                path = row[col_name]
                if not path:
                    continue
                file_path = Path(path)
                if file_path.exists():
                    files.append(
                        {
                            'id': file_id,
                            'name': file_path.name,
                            'size': file_path.stat().st_size,
                            'type': file_type,
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
def upload_video(file: UploadFile = File(...)):  # noqa: B008
    if not any(file.content_type.startswith(vtype) for vtype in ALLOWED_TYPES['video']):
        raise HTTPException(400, 'Invalid video format')

    file_path: Path | None = None
    try:
        file_path = upload_utils.save_upload(file, TEMP_DIR)
        full_path = upload_utils.normalize_path(file_path)
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
def upload_audio(file: UploadFile = File(...)):  # noqa: B008
    if not any(file.content_type.startswith(atype) for atype in ALLOWED_TYPES['audio']):
        raise HTTPException(400, 'Invalid audio format')

    file_path: Path | None = None
    try:
        file_path = upload_utils.save_upload(file, TEMP_DIR)
        normalized_path = upload_utils.normalize_path(file_path)
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


def get_answer(question: str) -> str:
    status = docs_table.insert([{'question': question}], return_rows=True)
    if not status.rows:
        return 'No response was generated. Please try asking another question.'
    row_uuid = status.rows[0]['uuid']
    result = docs_table.select(docs_table.answer).where(docs_table.uuid == row_uuid).collect()
    if len(result) == 0:
        return 'No response was generated. Please try asking another question.'
    answer = result['answer'][0]
    if not answer or answer.strip() == '':
        return 'An empty response was generated. Please try asking another question.'
    return answer


@router.post('/api/chat')
def chat(message: ChatMessage):
    try:
        conversations.insert([{'role': 'user', 'content': message.message, 'timestamp': datetime.now()}])
        response = get_answer(message.message)
        conversations.insert([{'role': 'assistant', 'content': response, 'timestamp': datetime.now()}])
        return JSONResponse(status_code=200, content={'response': response, 'used_files': []})
    except Exception as e:
        logger.error(f'Error in chat: {e!s}')
        raise HTTPException(500, str(e)) from e


@router.get('/')
def root():
    return 'Pixeltable Multimodal API'


@router.get('/health')
def health_check():
    return {'status': 'ok'}

"""Shared upload helpers for FastAPI showcase apps.

Use sync ``def`` route handlers with these helpers. Pixeltable I/O is synchronous;
FastAPI runs ``def`` endpoints on a thread pool (see operations.mdx).
"""

from __future__ import annotations

import atexit
import io
import os
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile


def create_upload_temp_dir() -> Path:
    temp_dir = Path(tempfile.mkdtemp())
    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
    return temp_dir


def safe_filename(filename: str | None) -> str:
    return Path(filename or 'upload').name


def save_upload(file: UploadFile, temp_dir: Path) -> Path:
    base = safe_filename(file.filename)
    suffix = Path(base).suffix
    stem = Path(base).stem or 'upload'
    temp_path = temp_dir / f'{stem}_{uuid4().hex}{suffix}'
    with temp_path.open('wb') as buffer:
        buffer.write(file.file.read())
    return temp_path


def read_text_query(upload: UploadFile | None) -> str:
    if upload is None:
        return ''
    body = upload.file.read()
    if body:
        return body.decode('utf-8', errors='replace').strip()
    return (upload.filename or '').strip()


def normalize_path(path: Path) -> str:
    return str(path.absolute()).replace(os.sep, '/')


def load_query_image(upload: UploadFile):
    import PIL.Image

    try:
        return PIL.Image.open(io.BytesIO(upload.file.read()))
    except PIL.UnidentifiedImageError as e:
        raise HTTPException(status_code=400, detail='Invalid image file') from e

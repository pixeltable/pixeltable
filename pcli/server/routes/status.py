import os
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter

from pcli.models import StatusResponse
from pcli.server.routes.health import _STARTED_AT

router = APIRouter()


def _dir_size(path: str | None) -> int | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    total = 0
    for entry in p.rglob('*'):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _redact_db_url(url: str | None) -> str | None:
    if url is None:
        return None
    try:
        return sa.make_url(url).render_as_string(hide_password=True)
    except Exception:
        return None


@router.get('/pcli/v0/status', response_model=StatusResponse)
def status(sizes: bool = False) -> StatusResponse:
    """Status snapshot. Pass `?sizes=1` to include media/file_cache disk usage (scans the directories)."""
    from pixeltable.dashboard import bridge

    s = bridge.get_status()
    cfg = s.get('config') or {}
    media_dir = cfg.get('media_dir')
    file_cache_dir = cfg.get('file_cache_dir')
    return StatusResponse(
        pxt_version=s['version'],
        pid=os.getpid(),
        started_at=_STARTED_AT,
        home=cfg.get('home'),
        db_url=_redact_db_url(cfg.get('db_url')),
        media_dir=media_dir,
        file_cache_dir=file_cache_dir,
        media_size_bytes=_dir_size(media_dir) if sizes else None,
        file_cache_size_bytes=_dir_size(file_cache_dir) if sizes else None,
        total_tables=s['total_tables'],
        total_errors=s['total_errors'],
    )

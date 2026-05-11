import datetime
import os

from fastapi import APIRouter

import pixeltable

from pcli.models import HealthResponse

router = APIRouter()
_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()


@router.get('/pcli/v0/health', response_model=HealthResponse)
def pcli_health() -> HealthResponse:
    return HealthResponse(ok=True, pxt_version=pixeltable.__version__, pid=os.getpid(), started_at=_STARTED_AT)


@router.get('/api/pixeltable-health')
def dashboard_health() -> dict:
    # compat for the existing dashboard probe
    return {'status': 'ok'}

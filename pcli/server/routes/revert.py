from fastapi import APIRouter, HTTPException

import pixeltable as pxt
from pixeltable import exceptions as excs

from pcli.models import RevertRequest, RevertResponse

router = APIRouter()


@router.post('/pcli/v0/revert', response_model=RevertResponse)
def revert(req: RevertRequest) -> RevertResponse:
    if req.steps < 1:
        raise HTTPException(400, 'steps must be >= 1')
    t = pxt.get_table(req.path)
    from_version = t.get_metadata()['version']
    try:
        for _ in range(req.steps):
            t.revert()
    except excs.Error as e:
        raise HTTPException(400, str(e))
    to_version = pxt.get_table(req.path).get_metadata()['version']
    return RevertResponse(path=req.path, from_version=from_version, to_version=to_version)

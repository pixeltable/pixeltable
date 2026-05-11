from fastapi import APIRouter, HTTPException

import pixeltable as pxt
from pixeltable import exceptions as excs

from pcli.models import MoveRequest, MoveResponse

router = APIRouter()


@router.post('/pcli/v0/move', response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    try:
        pxt.move(req.path, req.new_path)
    except excs.Error as e:
        raise HTTPException(400, str(e))
    return MoveResponse(path=req.path, new_path=req.new_path)

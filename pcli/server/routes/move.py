from fastapi import APIRouter

import pixeltable as pxt
from pcli.models import MoveRequest, MoveResponse

router = APIRouter()


@router.post('/pcli/v0/move', response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    pxt.move(req.path, req.new_path)
    return MoveResponse(path=req.path, new_path=req.new_path)

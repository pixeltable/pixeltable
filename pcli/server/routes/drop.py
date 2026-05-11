from fastapi import APIRouter, HTTPException

import pixeltable as pxt
from pixeltable import exceptions as excs

from pcli.models import DropRequest, DropResponse

router = APIRouter()


@router.post('/pcli/v0/drop', response_model=DropResponse)
def drop(req: DropRequest) -> DropResponse:
    try:
        if req.is_dir:
            pxt.drop_dir(req.path, force=req.cascade)
        else:
            pxt.drop_table(req.path, force=req.cascade)
    except excs.Error as e:
        raise HTTPException(400, str(e))
    return DropResponse(path=req.path, dropped=True)

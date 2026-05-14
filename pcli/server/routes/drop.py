from fastapi import APIRouter

import pixeltable as pxt

from pcli.models import DropRequest, DropResponse

router = APIRouter()


@router.post('/pcli/v0/drop', response_model=DropResponse)
def drop(req: DropRequest) -> DropResponse:
    if req.is_dir:
        pxt.drop_dir(req.path, force=req.cascade)
    else:
        pxt.drop_table(req.path, force=req.cascade)
    return DropResponse(path=req.path, dropped=True)

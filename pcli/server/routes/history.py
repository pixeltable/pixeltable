from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt
from pcli.models import HistoryRequest, HistoryResponse

router = APIRouter()


@router.post('/pcli/v0/history', response_model=HistoryResponse)
def history(req: HistoryRequest) -> HistoryResponse:
    versions = pxt.get_table(req.path).get_versions(req.n)
    return HistoryResponse(versions=jsonable_encoder(versions))

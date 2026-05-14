from fastapi import APIRouter

import pixeltable as pxt
from pcli.models import CountRequest, CountResponse

router = APIRouter()


@router.post('/pcli/v0/count', response_model=CountResponse)
def count(req: CountRequest) -> CountResponse:
    n = pxt.get_table(req.path).count()
    return CountResponse(path=req.path, count=n)

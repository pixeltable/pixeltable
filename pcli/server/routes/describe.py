from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt
from pcli.models import DescribeRequest, DescribeResponse

router = APIRouter()


@router.post('/pcli/v0/describe', response_model=DescribeResponse)
def describe(req: DescribeRequest) -> DescribeResponse:
    t = pxt.get_table(req.path)
    return DescribeResponse(text=repr(t), metadata=jsonable_encoder(t.get_metadata()))

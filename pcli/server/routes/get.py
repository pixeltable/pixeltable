import PIL.Image
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt

from pcli.models import GetRequest, GetResponse

router = APIRouter()


@router.post('/pcli/v0/get', response_model=GetResponse)
def get(req: GetRequest) -> GetResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()
    pk_names = md.get('primary_key')
    if not pk_names:
        raise HTTPException(400, f'{req.path}: no primary key declared; pcli get requires one')
    if len(req.pk) != len(pk_names):
        raise HTTPException(400, f'{req.path}: expected {len(pk_names)} PK value(s) for {pk_names}, got {len(req.pk)}')

    where = None
    for name, val in zip(pk_names, req.pk):
        cond = (t[name] == val)
        where = cond if where is None else where & cond
    result = t.where(where).limit(2).collect()
    if len(result) == 0:
        return GetResponse(pk_columns=pk_names, row=None)
    if len(result) > 1:
        raise HTTPException(500, f'{req.path}: {len(pk_names)}-column PK match returned multiple rows; catalog corruption?')

    r = result[0]
    columns = list(md['columns'].keys())
    row: dict = {}
    for c in columns:
        v = r[c]
        if isinstance(v, PIL.Image.Image):
            row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
        else:
            row[c] = v
    return GetResponse(pk_columns=pk_names, row=jsonable_encoder(row))

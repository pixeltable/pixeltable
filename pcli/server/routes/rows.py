import PIL.Image
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt
from pcli.models import RowsRequest, RowsResponse

router = APIRouter()


@router.post('/pcli/v0/rows', response_model=RowsResponse)
def rows(req: RowsRequest) -> RowsResponse:
    if req.n <= 0:
        raise HTTPException(400, 'n must be > 0')
    t = pxt.get_table(req.path)
    cols_md = t.get_metadata()['columns']
    if req.cols is not None:
        missing = [c for c in req.cols if c not in cols_md]
        if missing:
            raise HTTPException(400, f'unknown columns: {",".join(missing)}')
        columns = list(req.cols)
    else:
        # Skip unstored computed columns by default: selecting one forces evaluation,
        # which can be slow or expensive (LLM calls, etc.). The user can opt in via --cols.
        columns = [name for name, c in cols_md.items() if c.get('is_stored', True)]
    result = t.select(*[t[c] for c in columns]).limit(req.n).collect()
    out_rows: list[dict] = []
    for r in result:
        row: dict = {}
        for c in columns:
            v = r[c]
            if isinstance(v, PIL.Image.Image):
                row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
            else:
                row[c] = v
        out_rows.append(row)
    return RowsResponse(columns=columns, rows=jsonable_encoder(out_rows))

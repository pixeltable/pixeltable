from fastapi import APIRouter, HTTPException

import pixeltable as pxt
from pcli.models import ErrorEntry, ErrorsRequest, ErrorsResponse

router = APIRouter()


@router.post('/pcli/v0/errors', response_model=ErrorsResponse)
def errors(req: ErrorsRequest) -> ErrorsResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()

    pk_names = md.get('primary_key')
    if not pk_names:
        raise HTTPException(400, f'{req.path}: no primary key declared; pcli errors requires one')

    computed = [name for name, c in md['columns'].items() if c['is_computed'] and (req.col is None or name == req.col)]
    if not computed:
        return ErrorsResponse(entries=[])

    where = None
    for col in computed:
        cond = t[col].errortype != None  # pxt overrides `!=` to return an Expr
        where = cond if where is None else where | cond

    select_args: list = [t[c] for c in pk_names]
    for col in computed:
        select_args += [t[col].errortype, t[col].errormsg]
    rows = t.where(where).select(*select_args).collect()

    entries = []
    for r in rows:
        pk = {name: r[name] for name in pk_names}
        for col in computed:
            etype = r[f'{col}_errortype']
            emsg = r[f'{col}_errormsg']
            if etype is not None:
                entries.append(ErrorEntry(pk=pk, column=col, errortype=etype, errormsg=emsg))
    return ErrorsResponse(entries=entries)

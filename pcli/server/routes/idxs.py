from typing import Any

from fastapi import APIRouter

import pixeltable as pxt
from pcli.models import IdxEntry, IdxsRequest, IdxsResponse

from .columns import _all_tables

router = APIRouter()


@router.post('/pcli/v0/idxs', response_model=IdxsResponse)
def idxs(req: IdxsRequest) -> IdxsResponse:
    paths = [req.path] if req.path else _all_tables()
    entries: list[IdxEntry] = []
    for path in paths:
        try:
            md = pxt.get_table(path).get_metadata()
        except Exception:
            continue
        for name, idx in md['indices'].items():
            if req.embedding_only and idx['index_type'] != 'embedding':
                continue
            params: Any = idx.get('parameters') or {}
            entries.append(
                IdxEntry(
                    table=path,
                    name=name,
                    columns=idx['columns'],
                    index_type=idx['index_type'],
                    metric=params.get('metric'),
                    embedding=params.get('embedding'),
                )
            )
    return IdxsResponse(entries=entries)

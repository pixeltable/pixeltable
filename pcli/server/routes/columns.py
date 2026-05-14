from typing import Any

from fastapi import APIRouter

import pixeltable as pxt
from pcli.models import ColumnEntry, ColumnsRequest, ColumnsResponse
from pixeltable import exceptions as excs

router = APIRouter()


@router.post('/pcli/v0/columns', response_model=ColumnsResponse)
def columns(req: ColumnsRequest) -> ColumnsResponse:
    paths = [req.path] if req.path else _all_tables()
    entries: list[ColumnEntry] = []
    for path in paths:
        try:
            md = pxt.get_table(path).get_metadata()
        except excs.Error:
            # skip tables whose metadata can't be loaded; other exceptions propagate as 500s
            continue
        for name, c in md['columns'].items():
            if req.computed_only and not c['is_computed']:
                continue
            entries.append(
                ColumnEntry(
                    table=path,
                    column=name,
                    is_computed=c['is_computed'],
                    type_=c['type_'],
                    computed_with=c['computed_with'],
                    depends_on=c['depends_on'],
                )
            )
    return ColumnsResponse(entries=entries)


def _all_tables() -> list[str]:
    paths: list[str] = []

    def walk(nodes: list[Any]) -> None:
        for n in nodes:
            if n['kind'] == 'directory':
                walk(n['entries'])
            else:
                paths.append(n['path'])

    walk(pxt.get_dir_tree())
    return paths

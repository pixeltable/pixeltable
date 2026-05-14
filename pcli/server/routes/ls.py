from fastapi import APIRouter

import pixeltable as pxt
from pixeltable import exceptions as excs

from pcli.models import LsEntry, LsRequest, LsResponse

router = APIRouter()


@router.post('/pcli/v0/ls', response_model=LsResponse)
def ls(req: LsRequest) -> LsResponse:
    tree = pxt.get_dir_tree()
    nodes = _descend(tree, req.path)
    if req.tree:
        return LsResponse(entries=[], tree={'path': req.path, 'entries': nodes})
    return LsResponse(entries=[_to_entry(n, req) for n in nodes])


def _descend(tree: list[dict], path: str) -> list[dict]:
    """Find children of `path` (`.` or `/` separated). Empty path = root."""
    parts = [p for p in path.replace('/', '.').split('.') if p]
    cur = tree
    for part in parts:
        for node in cur:
            if node['name'] == part and node['kind'] == 'directory':
                cur = node['entries']
                break
        else:
            raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f"Path '{path}' does not exist.")
    return cur


def _to_entry(node: dict, req: LsRequest) -> LsEntry:
    # pxt kinds: 'directory' | 'table' | 'view' | 'snapshot' | 'replica' (see pixeltable.types.TableKind)
    kind = 'dir' if node['kind'] == 'directory' else node['kind']
    entry = LsEntry(path=node['path'], kind=kind)
    if kind != 'dir':
        entry.last_version = node.get('version')
        tbl = pxt.get_table(node['path'])
        md = tbl.get_metadata()
        cols = md.get('columns') or {}
        idxs = md.get('indices') or {}
        entry.num_cols = len(cols)
        entry.flags = ('c' if any(c.get('is_computed') for c in cols.values()) else '') + ('i' if idxs else '')
        if req.counts:
            entry.num_rows = tbl.count()
    return entry

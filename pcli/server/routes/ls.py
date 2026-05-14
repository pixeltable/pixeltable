from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter

import pixeltable as pxt
from pcli.models import LsEntry, LsRequest, LsResponse
from pixeltable import exceptions as excs

router = APIRouter()

# Cap for the row-count thread pool. count() is SQL-bound (GIL-released during the query),
# so going wider than the DB can serve in parallel just queues. 16 is comfortably below
# pixeltable's default connection-pool ceiling.
_COUNT_POOL_WORKERS = 16


@router.post('/pcli/v0/ls', response_model=LsResponse)
def ls(req: LsRequest) -> LsResponse:
    tree = pxt.get_dir_tree()
    nodes = _descend(tree, req.path)
    if req.tree:
        return LsResponse(entries=[], tree={'path': req.path, 'entries': nodes})

    entries = [_to_entry(n, details=req.details) for n in nodes]
    if req.counts:
        _fill_counts(entries)
    return LsResponse(entries=entries)


def _descend(tree: list[Any], path: str) -> list[Any]:
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


def _to_entry(node: Any, details: bool) -> LsEntry:
    # pxt kinds: 'directory' | 'table' | 'view' | 'snapshot' | 'replica' (see pixeltable.types.TableKind)
    kind = 'dir' if node['kind'] == 'directory' else node['kind']
    entry = LsEntry(path=node['path'], kind=kind)
    if kind != 'dir':
        entry.last_version = node.get('version')
        if details:
            md = pxt.get_table(node['path']).get_metadata()
            cols = md.get('columns') or {}
            idxs = md.get('indices') or {}
            entry.num_cols = len(cols)
            entry.flags = ('c' if any(c.get('is_computed') for c in cols.values()) else '') + ('i' if idxs else '')
    return entry


def _fill_counts(entries: list[LsEntry]) -> None:
    # Resolve and count in the worker thread: pxt.get_table() returns a thread-local
    # handle that raises if reused across threads.
    targets = [e for e in entries if e.kind != 'dir']
    if not targets:
        return
    paths = [e.path for e in targets]
    with ThreadPoolExecutor(max_workers=min(_COUNT_POOL_WORKERS, len(targets))) as pool:
        for e, n in zip(targets, pool.map(_safe_count, paths)):
            e.num_rows = n


def _safe_count(path: str) -> int | None:
    try:
        return pxt.get_table(path).count()
    except excs.Error:
        return None

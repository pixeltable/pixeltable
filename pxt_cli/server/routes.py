import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import PIL.Image
import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, Query
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.types import TreeNode
from pxt_cli import models, probe

router = APIRouter()
_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()
# Freeze the identity fingerprint at import time so /health reports what the daemon was
# launched with, not what os.environ looks like right now. Used to trigger a daemon restart.
_IDENTITY: dict[str, Any] = probe.identity()


# Path validation: paths in URLs come straight from the FastAPI {path:path} converter, which
# accepts anything. Reject the same shapes the request-model validator used to reject so a
# bad path gives a clear 400 instead of bubbling out of pixeltable as a generic error.
def _validate_path(path: str) -> str:
    if path == '':
        return path
    if '.' in path:
        raise HTTPException(400, f"path uses '/' as the separator; got {path!r}")
    if path.endswith('/'):
        raise HTTPException(400, f"path must not end with '/'; got {path!r}")
    if '//' in path:
        raise HTTPException(400, f"path must not contain empty components ('//'); got {path!r}")
    return path


@router.get('/api/health')
def health() -> models.HealthResponse:
    return models.HealthResponse(ok=True, pid=os.getpid(), started_at=_STARTED_AT, **_IDENTITY)


@router.get('/api/status')
def status(sizes: bool = False) -> models.StatusResponse:
    from pixeltable.dashboard import bridge

    s = bridge.get_status()
    cfg = s['config']
    media_dir = cfg.get('media_dir')
    file_cache_dir = cfg.get('file_cache_dir')
    return models.StatusResponse(
        pxt_version=s['version'],
        pid=os.getpid(),
        started_at=_STARTED_AT,
        home=cfg.get('home'),
        db_url=_redact_db_password(cfg.get('db_url')),
        media_dir=media_dir,
        file_cache_dir=file_cache_dir,
        media_size_bytes=_dir_size(media_dir) if sizes else None,
        file_cache_size_bytes=_dir_size(file_cache_dir) if sizes else None,
        total_tables=s['total_tables'],
        total_errors=s['total_errors'],
    )


@router.get('/api/config')
def config() -> models.ConfigResponse:
    # Two-layer redaction so a new sensitive key never silently leaks:
    #   1. params from registered API client factories (eg openai.api_key, replicate.api_token)
    #   2. name-suffix match for keys that aren't tied to an Env-registered client
    #      (eg azure.storage_account_key, anything ending in _password)
    client_creds: set[str] = {p for params in Env.get().get_client_credential_params().values() for p in params}
    sensitive_suffixes = ('_key', '_token', '_secret', '_password')
    entries: list[models.ConfigEntry] = []
    for ck in Config.get().config_keys():
        source = Config.get().get_value_source(ck.key, section=ck.section)
        is_sensitive = ck.key in client_creds or any(ck.key.endswith(s) for s in sensitive_suffixes)
        if source == 'unset':
            value: str | None = None
        elif is_sensitive:
            value = '<redacted>'
        else:
            raw: Any = Config.get().get_value(ck.key, ck.expected_type, section=ck.section)
            value = None if raw is None else str(raw)
        entries.append(
            models.ConfigEntry(
                section=ck.section,
                key=ck.key,
                value=value,
                source=source,
                description=ck.description,
                expected_type=getattr(ck.expected_type, '__name__', str(ck.expected_type)),
            )
        )
    return models.ConfigResponse(config_file=str(Config.get().config_file), entries=entries)


# --- directories ------------------------------------------------------------------------------


@router.get('/api/dirs')
def list_root(tree: bool = False, details: bool = False, counts: bool = False) -> models.LsResponse:
    return _list_dir('', tree=tree, details=details, counts=counts)


@router.get('/api/dirs/{path:path}')
def list_dir(path: str, tree: bool = False, details: bool = False, counts: bool = False) -> models.LsResponse:
    _validate_path(path)
    return _list_dir(path, tree=tree, details=details, counts=counts)


def _list_dir(path: str, *, tree: bool, details: bool, counts: bool) -> models.LsResponse:
    full_tree = pxt.get_dir_tree()
    nodes = _get_dir_children(full_tree, path)
    if tree:
        return models.LsResponse(entries=[], tree={'path': path, 'entries': nodes})
    entries = [_to_entry(n, details=details) for n in nodes]
    if counts:
        _fill_counts(entries)
    return models.LsResponse(entries=entries)


# --- table-scoped reads (specific suffixes BEFORE the catch-all describe) ----------------------


@router.get('/api/tables/{path:path}/rows')
def table_rows(path: str, n: int = Query(default=10, ge=1, le=1000), cols: str | None = None) -> models.RowsResponse:
    _validate_path(path)
    cols_list = _split_csv(cols)
    if cols_list is not None and len(cols_list) > 1000:
        raise HTTPException(400, 'too many columns requested (max 1000)')

    t = pxt.get_table(path)
    cols_md = t.get_metadata()['columns']
    if cols_list is not None:
        missing = [c for c in cols_list if c not in cols_md]
        if len(missing) > 0:
            raise HTTPException(400, f'unknown columns: {",".join(missing)}')
        columns_list = list(cols_list)
    else:
        # skip unstored computed columns unless the user explicitly asked for them: evaluation can take
        # an unbounded amount of time
        columns_list = [name for name, c in cols_md.items() if c.get('is_stored', True)]
    result = t.select(*[t[c] for c in columns_list]).limit(n).collect()
    out_rows: list[dict[str, Any]] = []
    for r in result:
        row: dict[str, Any] = {}
        for c in columns_list:
            v = r[c]
            if isinstance(v, PIL.Image.Image):
                row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
            else:
                row[c] = v
        out_rows.append(row)
    return models.RowsResponse(columns=columns_list, rows=jsonable_encoder(out_rows))


@router.get('/api/tables/{path:path}/row')
def table_row(
    path: str,
    pk: list[str] = Query(..., description='PK values in column order; repeat ?pk= per value'),
    cols: str | None = None,
) -> models.GetResponse:
    _validate_path(path)
    if len(pk) == 0:
        raise HTTPException(400, "missing or empty 'pk' query parameter")
    # PK values arrive as strings over HTTP; coerce numeric-looking ones to int/float so a
    # PK column typed as Int compares correctly. String-typed PK columns whose values look
    # like numbers (eg the string '42') are a documented limitation - there's no way to
    # force a string interpretation from the URL.
    pk_values: list[Any] = [_coerce_pk(v) for v in pk]
    cols_list = _split_csv(cols)

    t = pxt.get_table(path)
    md = t.get_metadata()
    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise HTTPException(400, f'{path}: no primary key declared; row lookup requires one')
    if len(pk_values) != len(pk_names):
        raise HTTPException(400, f'{path}: expected {len(pk_names)} PK value(s) for {pk_names}, got {len(pk_values)}')

    cols_md = md['columns']
    if cols_list is not None:
        missing = [c for c in cols_list if c not in cols_md]
        if len(missing) > 0:
            raise HTTPException(400, f'unknown columns: {",".join(missing)}')
        cols_to_fetch = list(cols_list)
    else:
        # skip unstored computed columns unless the user explicitly asked for them: evaluation can take
        # an unbounded amount of time
        cols_to_fetch = [name for name, c in cols_md.items() if c.get('is_stored', True)]

    where = None
    for name, val in zip(pk_names, pk_values):
        cond = t[name] == val
        where = cond if where is None else where & cond
    result = t.where(where).select(*[t[c] for c in cols_to_fetch]).limit(2).collect()
    if len(result) == 0:
        return models.GetResponse(pk_columns=pk_names, row=None)
    if len(result) > 1:
        raise HTTPException(500, f'{path}: {len(pk_names)}-column PK match returned multiple rows; catalog corruption?')

    r = result[0]
    row: dict[str, Any] = {}
    for c in cols_to_fetch:
        v = r[c]
        if isinstance(v, PIL.Image.Image):
            row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
        else:
            row[c] = v
    return models.GetResponse(pk_columns=pk_names, row=jsonable_encoder(row))


@router.get('/api/tables/{path:path}/count')
def table_count(path: str) -> models.CountResponse:
    _validate_path(path)
    return models.CountResponse(path=path, count=pxt.get_table(path).count())


@router.get('/api/tables/{path:path}/errors')
def table_errors(path: str, col: str | None = None) -> models.ErrorsResponse:
    _validate_path(path)
    t = pxt.get_table(path)
    md = t.get_metadata()

    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise HTTPException(400, f'{path}: no primary key declared; errors view requires one')

    # Validate col up front so a typo'd --col returns a 400 instead of a misleading
    # empty result; matches the unknown-column behavior of /rows and /row.
    if col is not None:
        col_md = md['columns'].get(col)
        if col_md is None:
            raise HTTPException(400, f'unknown column: {col}')
        if not (col_md['is_computed'] and col_md.get('is_stored')):
            raise HTTPException(400, f'{col}: not a stored computed column (errors view only addresses those)')

    # errortype/errormsg only apply to stored computed columns
    computed = [
        name
        for name, c in md['columns'].items()
        if c['is_computed'] and c.get('is_stored') and (col is None or name == col)
    ]
    if len(computed) == 0:
        return models.ErrorsResponse(entries=[])

    where = None
    for c_name in computed:
        cond = t[c_name].errortype != None  # pxt overrides != to return an Expr
        where = cond if where is None else where | cond

    select_args: list[Any] = [t[c] for c in pk_names]
    for c_name in computed:
        select_args += [t[c_name].errortype, t[c_name].errormsg]
    rows = t.where(where).select(*select_args).collect()

    entries: list[models.ErrorEntry] = []
    for r in rows:
        pk = {name: r[name] for name in pk_names}
        for c_name in computed:
            etype = r[f'{c_name}_errortype']
            emsg = r[f'{c_name}_errormsg']
            if etype is not None:
                entries.append(models.ErrorEntry(pk=pk, column=c_name, errortype=etype, errormsg=emsg))
    return models.ErrorsResponse(entries=entries)


@router.get('/api/tables/{path:path}/history')
def table_history(path: str, n: int | None = None) -> models.HistoryResponse:
    _validate_path(path)
    versions = pxt.get_table(path).get_versions(n)
    return models.HistoryResponse(versions=jsonable_encoder(versions))


# --- table-scoped mutators (POST) ------------------------------------------------------------


@router.post('/api/tables/{path:path}/drop')
def drop_table(path: str, req: models.DropBody) -> models.DropResponse:
    _validate_path(path)
    pxt.drop_table(path, force=req.cascade)
    return models.DropResponse(path=path, dropped=True)


@router.post('/api/tables/{path:path}/revert')
def revert(path: str, req: models.RevertBody) -> models.RevertResponse:
    _validate_path(path)
    if req.steps < 1:
        raise HTTPException(400, 'steps must be >= 1')
    t = pxt.get_table(path)
    from_version = t.get_metadata()['version']
    if req.steps > from_version:
        raise HTTPException(400, f'cannot revert {req.steps} step(s): {path} is at version {from_version}')
    for _ in range(req.steps):
        t.revert()
    to_version = pxt.get_table(path).get_metadata()['version']
    return models.RevertResponse(path=path, from_version=from_version, to_version=to_version)


# --- table describe (catch-all, registered LAST so /rows, /count, etc. win) ------------------


@router.get('/api/tables/{path:path}')
def describe_table(path: str) -> models.DescribeResponse:
    _validate_path(path)
    t = pxt.get_table(path)
    return models.DescribeResponse(text=repr(t), metadata=jsonable_encoder(t.get_metadata()))


# --- cross-table aggregates --------------------------------------------------------------------


@router.get('/api/columns')
def columns(path: str | None = None, computed: bool = False) -> models.ColumnsResponse:
    if path is not None:
        _validate_path(path)
    paths = [path] if path is not None and path != '' else _collect_table_paths()
    entries: list[models.ColumnEntry] = []
    for p in paths:
        try:
            md = pxt.get_table(p).get_metadata()
        except excs.NotFoundError:
            # a table can disappear between dir-tree traversal and metadata fetch; other
            # pixeltable errors propagate as 500s so real bugs surface
            continue
        for name, c in md['columns'].items():
            if computed and not c['is_computed']:
                continue
            entries.append(
                models.ColumnEntry(
                    table=p,
                    column=name,
                    is_computed=c['is_computed'],
                    type_=c['type_'],
                    computed_with=c['computed_with'],
                    depends_on=c['depends_on'],
                )
            )
    return models.ColumnsResponse(entries=entries)


@router.get('/api/indexes')
def indexes(path: str | None = None, embedding: bool = False) -> models.IdxsResponse:
    if path is not None:
        _validate_path(path)
    paths = [path] if path is not None and path != '' else _collect_table_paths()
    entries: list[models.IdxEntry] = []
    for p in paths:
        try:
            md = pxt.get_table(p).get_metadata()
        except excs.NotFoundError:
            continue
        for name, idx in md['indices'].items():
            if embedding and idx['index_type'] != 'embedding':
                continue
            params = idx.get('parameters')
            entries.append(
                models.IdxEntry(
                    table=p,
                    name=name,
                    columns=idx['columns'],
                    index_type=idx['index_type'],
                    metric=params['metric'] if params is not None else None,
                    embedding=params['embedding'] if params is not None else None,
                )
            )
    return models.IdxsResponse(entries=entries)


# --- directory mutators + cross-cutting move --------------------------------------------------


@router.post('/api/dirs/{path:path}/drop')
def drop_dir(path: str, req: models.DropBody) -> models.DropResponse:
    _validate_path(path)
    pxt.drop_dir(path, force=req.cascade)
    return models.DropResponse(path=path, dropped=True)


@router.post('/api/move')
def move(req: models.MoveBody) -> models.MoveResponse:
    pxt.move(req.path, req.new_path)
    return models.MoveResponse(path=req.path, new_path=req.new_path)


# --- helpers (unchanged) ---------------------------------------------------------------------


# count() is SQL-bound (GIL-released during the query), so going wider than the DB can
# serve in parallel just queues. 16 is comfortably below pixeltable's default connection-pool ceiling.
_COUNT_POOL_WORKERS = 16


def _coerce_pk(s: str) -> Any:
    """Numeric-looking PK strings become int or float; everything else stays a string.
    Mirrors the previous client-side coercion now that PK values arrive untyped over HTTP."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _split_csv(s: str | None) -> list[str] | None:
    """Comma-separated query strings: 'a,b,c' -> ['a', 'b', 'c']. Reject empty tokens
    ('a,,b') so the server doesn't silently drop a malformed value."""
    if s is None:
        return None
    parts = [p.strip() for p in s.split(',')]
    if any(p == '' for p in parts):
        raise HTTPException(400, f'comma-separated value contains an empty token: {s!r}')
    return parts


def _get_dir_children(tree: list[TreeNode], dir_path: str) -> list[TreeNode]:
    """Find children of dir_path (empty path = root)."""
    parts = dir_path.split('/') if dir_path else []
    cur = tree
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        match = next((n for n in cur if n['name'] == part), None)
        if match is None:
            raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f'Path {dir_path!r} does not exist.')
        if match['kind'] != 'directory':
            joined = '/'.join(traversed)
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, f'{joined!r} is a {match["kind"]}, not a directory'
            )
        cur = match['entries']
    return cur


def _to_entry(node: TreeNode, details: bool) -> models.LsEntry:
    # pxt kinds: 'directory' | 'table' | 'view' | 'snapshot' | 'replica' (see pixeltable.types.TableKind)
    if node['kind'] == 'directory':
        return models.LsEntry(path=node['path'], kind='dir')
    entry = models.LsEntry(path=node['path'], kind=node['kind'], last_version=node['version'])
    if details:
        md = pxt.get_table(node['path']).get_metadata()
        cols = md['columns']
        indices = md['indices']
        entry.num_cols = len(cols)
        has_computed = any(c['is_computed'] for c in cols.values())
        entry.flags = ('c' if has_computed else '') + ('i' if len(indices) > 0 else '')
    return entry


def _fill_counts(entries: list[models.LsEntry]) -> None:
    # Resolve and count in the worker thread: pxt.get_table() returns a thread-local handle
    # that raises if reused across threads.
    targets = [e for e in entries if e.kind != 'dir']
    if len(targets) == 0:
        return
    paths = [e.path for e in targets]
    with ThreadPoolExecutor(max_workers=min(_COUNT_POOL_WORKERS, len(targets))) as pool:
        for e, n in zip(targets, pool.map(_tbl_count, paths)):
            e.num_rows = n


def _tbl_count(path: str) -> int | None:
    try:
        return pxt.get_table(path).count()
    except excs.Error:
        return None


def _collect_table_paths() -> list[str]:
    paths: list[str] = []

    def collect(nodes: list[TreeNode]) -> None:
        for n in nodes:
            if n['kind'] == 'directory':
                collect(n['entries'])
            else:
                paths.append(n['path'])

    collect(pxt.get_dir_tree())
    return paths


def _dir_size(path: str | None) -> int | None:
    if path is None or not os.path.isdir(path):
        return None
    total = 0
    # os.walk + onerror gives best-effort traversal: permission errors or files vanishing
    # mid-scan are swallowed rather than turning ?sizes=1 into a 500.
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def _redact_db_password(url: str | None) -> str | None:
    """Replace the password in a SQLAlchemy URL with '***'. Returns None if the URL can't be parsed."""
    if url is None:
        return None
    try:
        return sa.make_url(url).render_as_string(hide_password=True)
    except Exception:
        return None

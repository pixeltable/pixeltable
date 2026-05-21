import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import PIL.Image
import sqlalchemy as sa

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.types import TreeNode
from pxt_cli import models
from pxt_cli.utils import identity

from . import bridge
from .router import RawResponse, Request, Router

router = Router()
_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()

# Freeze the identity fingerprint at import time so /health reports what the daemon was
# launched with, not what os.environ looks like right now. Used to trigger a daemon restart.
_IDENTITY: dict[str, Any] = identity()


def _validate_path(path: str) -> str:
    """Reject URL paths whose shape pixeltable would later reject with a generic error.

    The router's {path:path} converter accepts anything between the static prefix and
    suffix, including the empty string and shapes pixeltable's own path parser refuses.
    """
    if path == '':
        return path
    # Reject ASCII control characters before any other check: the path is later interpolated
    # into response headers (eg Content-Disposition in dashboard_table_export), so CR/LF/etc.
    # in a URL-decoded path would otherwise enable response splitting / header injection.
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'path contains control characters; got {path!r}')
    if '.' in path:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"path uses '/' as the separator; got {path!r}")
    if path.endswith('/'):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f"path must not end with '/'; got {path!r}")
    if '//' in path:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f"path must not contain empty components ('//'); got {path!r}"
        )
    return path


@router.get('/api/health')
def health(_req: Request) -> models.HealthResponse:
    return models.HealthResponse(ok=True, pid=os.getpid(), started_at=_STARTED_AT, **_IDENTITY)


@router.get('/api/status')
def status(req: Request) -> models.StatusResponse:
    sizes = req.query_bool('sizes')
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
def config(_req: Request) -> models.ConfigResponse:
    # Two-layer redaction so a new sensitive key never silently leaks:
    # - params from registered API client factories
    # - name-suffix match for keys that aren't tied to an Env-registered client (eg, azure.storage_account_key)
    client_creds: set[str] = set(Env.get().get_client_credential_params().values())
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
                source=str(source),
                description=ck.description,
                expected_type=getattr(ck.expected_type, '__name__', str(ck.expected_type)),
            )
        )
    return models.ConfigResponse(config_file=str(Config.get().config_file), entries=entries)


@router.get('/api/dirs')
def list_root(req: Request) -> models.LsResponse:
    return _list_dir(
        '', tree=req.query_bool('tree'), details=req.query_bool('details'), counts=req.query_bool('counts')
    )


@router.get('/api/dirs/{path:path}')
def list_dir(req: Request) -> models.LsResponse:
    path = _validate_path(req.path_params['path'])
    return _list_dir(
        path, tree=req.query_bool('tree'), details=req.query_bool('details'), counts=req.query_bool('counts')
    )


def _list_dir(path: str, *, tree: bool, details: bool, counts: bool) -> models.LsResponse:
    full_tree = pxt.get_dir_tree()
    nodes = _get_dir_children(full_tree, path)
    if tree:
        return models.LsResponse(entries=[], tree={'path': path, 'entries': nodes})
    entries = [_to_entry(n, details=details) for n in nodes]
    if counts:
        _fill_counts(entries)
    return models.LsResponse(entries=entries)


@router.get('/api/tables/{path:path}/rows')
def table_rows(req: Request) -> models.RowsResponse:
    path = _validate_path(req.path_params['path'])
    n = req.query_int('n', default=10, ge=1, le=1000)
    cols_list = _split_csv(req.query_str('cols'))
    if cols_list is not None and len(cols_list) > 1000:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'too many columns requested (max 1000)')

    t = pxt.get_table(path)
    cols_md = t.get_metadata()['columns']
    if cols_list is not None:
        missing = [c for c in cols_list if c not in cols_md]
        if len(missing) > 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'unknown columns: {",".join(missing)}')
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
    return models.RowsResponse(columns=columns_list, rows=out_rows)


@router.get('/api/tables/{path:path}/row')
def table_row(req: Request) -> models.GetResponse:
    path = _validate_path(req.path_params['path'])
    pk = req.query_list('pk')
    if len(pk) == 0:
        raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, "missing or empty 'pk' query parameter")
    # PK values arrive as strings over HTTP; coerce numeric-looking ones to int/float so a
    # PK column typed as Int compares correctly. String-typed PK columns whose values look
    # like numbers (eg the string '42') are a documented limitation - there's no way to
    # force a string interpretation from the URL.
    pk_values: list[Any] = [_coerce_pk(v) for v in pk]
    cols_list = _split_csv(req.query_str('cols'))

    t = pxt.get_table(path)
    md = t.get_metadata()
    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'{path}: no primary key declared; row lookup requires one'
        )
    if len(pk_values) != len(pk_names):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'{path}: expected {len(pk_names)} PK value(s) for {pk_names}, got {len(pk_values)}',
        )

    cols_md = md['columns']
    if cols_list is not None:
        missing = [c for c in cols_list if c not in cols_md]
        if len(missing) > 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'unknown columns: {",".join(missing)}')
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
        raise RuntimeError(f'{path}: {len(pk_names)}-column PK match returned multiple rows; catalog corruption?')

    r = result[0]
    row: dict[str, Any] = {}
    for c in cols_to_fetch:
        v = r[c]
        if isinstance(v, PIL.Image.Image):
            row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
        else:
            row[c] = v
    return models.GetResponse(pk_columns=pk_names, row=row)


@router.get('/api/tables/{path:path}/count')
def table_count(req: Request) -> models.CountResponse:
    path = _validate_path(req.path_params['path'])
    return models.CountResponse(path=path, count=pxt.get_table(path).count())


@router.get('/api/tables/{path:path}/errors')
def table_errors(req: Request) -> models.ErrorsResponse:
    path = _validate_path(req.path_params['path'])
    col = req.query_str('col')
    t = pxt.get_table(path)
    md = t.get_metadata()

    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'{path}: no primary key declared; errors view requires one'
        )

    if col is not None:
        col_md = md['columns'].get(col)
        if col_md is None:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'unknown column: {col}')
        if not (col_md['is_computed'] and col_md.get('is_stored')):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'{col}: not a stored computed column (errors view only addresses those)',
            )

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
def table_history(req: Request) -> models.HistoryResponse:
    path = _validate_path(req.path_params['path'])
    n = req.query_int('n', default=None, ge=1)
    versions = pxt.get_table(path).get_versions(n)
    return models.HistoryResponse(versions=[dict(v) for v in versions])


@router.post('/api/tables/{path:path}/drop')
def drop_table(req: Request) -> models.DropResponse:
    path = _validate_path(req.path_params['path'])
    body = req.body(models.DropBody)
    pxt.drop_table(path, force=body.cascade)
    return models.DropResponse(path=path, dropped=True)


@router.post('/api/tables/{path:path}/revert')
def revert(req: Request) -> models.RevertResponse:
    path = _validate_path(req.path_params['path'])
    body = req.body(models.RevertBody)
    if body.steps < 1:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, 'steps must be >= 1')
    t = pxt.get_table(path)
    from_version = t.get_metadata()['version']
    if body.steps > from_version:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'cannot revert {body.steps} step(s): {path} is at version {from_version}'
        )
    for _ in range(body.steps):
        t.revert()
    to_version = pxt.get_table(path).get_metadata()['version']
    return models.RevertResponse(path=path, from_version=from_version, to_version=to_version)


@router.get('/api/tables/{path:path}')
def describe_table(req: Request) -> models.DescribeResponse:
    path = _validate_path(req.path_params['path'])
    t = pxt.get_table(path)
    return models.DescribeResponse(text=repr(t), metadata=dict(t.get_metadata()))


@router.get('/api/columns')
def columns(req: Request) -> models.ColumnsResponse:
    path = req.query_str('path')
    computed = req.query_bool('computed')
    # An empty `?path=` is almost always an interpolated variable resolving to empty;
    # treat it as a bad request rather than silently dumping the full catalog.
    if path == '':
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, "'path' query parameter is empty; omit it entirely for all tables"
        )
    if path is not None:
        _validate_path(path)
    paths = [path] if path is not None else _collect_table_paths()
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
def indexes(req: Request) -> models.IdxsResponse:
    path = req.query_str('path')
    embedding = req.query_bool('embedding')
    if path == '':
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, "'path' query parameter is empty; omit it entirely for all tables"
        )
    if path is not None:
        _validate_path(path)
    paths = [path] if path is not None else _collect_table_paths()
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


@router.post('/api/dirs/{path:path}/drop')
def drop_dir(req: Request) -> models.DropResponse:
    path = _validate_path(req.path_params['path'])
    body = req.body(models.DropBody)
    pxt.drop_dir(path, force=body.cascade)
    return models.DropResponse(path=path, dropped=True)


@router.post('/api/move')
def move(req: Request) -> models.MoveResponse:
    body = req.body(models.MoveBody)
    pxt.move(body.path, body.new_path)
    return models.MoveResponse(path=body.path, new_path=body.new_path)


@router.get('/api/dashboard/search')
def dashboard_search(req: Request) -> dict[str, Any]:
    q = req.query_str('q', default='') or ''
    limit = req.query_int('limit', default=50, ge=1, le=100)
    if q == '':
        return {'query': '', 'directories': [], 'tables': [], 'columns': []}
    return bridge.search(q, limit=limit)


@router.get('/api/dashboard/tables/{path:path}/meta')
def dashboard_table_meta(req: Request) -> dict[str, Any]:
    path = _validate_path(req.path_params['path'])
    return dict(pxt.get_table(path).get_metadata())


@router.get('/api/dashboard/tables/{path:path}/pipeline')
def dashboard_pipeline(req: Request) -> dict[str, Any]:
    path = _validate_path(req.path_params['path'])
    return bridge.get_pipeline(tbl_path=path)


@router.get('/api/dashboard/pipeline')
def dashboard_pipeline_root(_req: Request) -> dict[str, Any]:
    return bridge.get_pipeline(tbl_path=None)


@router.get('/api/dashboard/tables/{path:path}/data')
def dashboard_table_data(req: Request) -> dict[str, Any]:
    path = _validate_path(req.path_params['path'])
    return bridge.get_table_data(
        path,
        offset=req.query_int('offset', default=0, ge=0),
        limit=req.query_int('limit', default=50, ge=1, le=500),
        order_by=req.query_str('order_by'),
        order_desc=req.query_bool('order_desc'),
        errors_only=req.query_bool('errors_only'),
    )


@router.get('/api/dashboard/tables/{path:path}/export')
def dashboard_table_export(req: Request) -> RawResponse:
    path = _validate_path(req.path_params['path'])
    limit = req.query_int('limit', default=100_000, ge=1, le=1_000_000)
    body = bridge.export_table_csv(path, limit=limit)
    filename = path.replace('/', '_') + '.csv'
    return RawResponse(
        body=body,
        content_type='text/csv; charset=utf-8',
        extra_headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


# count() is SQL-bound (GIL-released during the query), so going wider than the DB can
# serve in parallel just queues. 16 is comfortably below pixeltable's default connection-pool ceiling.
_COUNT_POOL_WORKERS = 16


def _coerce_pk(s: str) -> Any:
    """Numeric-looking PK strings become int or float; everything else stays a string.

    PK values arrive untyped over HTTP, so we restore their natural type here.
    """
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
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'comma-separated value contains an empty token: {s!r}'
        )
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

"""All pcli HTTP routes in a single module.

Each route is registered on the shared `router` and mounted at `/pcli/v0/*`. Module-private
helpers (`_*`) live next to the routes that use them; route handlers consume them directly
rather than crossing module boundaries.
"""

import datetime
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import PIL.Image
import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

import pixeltable
import pixeltable as pxt
from pcli._paths import redact_home
from pcli.models import (
    ColumnEntry,
    ColumnsRequest,
    ColumnsResponse,
    CountRequest,
    CountResponse,
    DescribeRequest,
    DescribeResponse,
    DropRequest,
    DropResponse,
    EnvResponse,
    ErrorEntry,
    ErrorsRequest,
    ErrorsResponse,
    GetRequest,
    GetResponse,
    HealthResponse,
    HistoryRequest,
    HistoryResponse,
    IdxEntry,
    IdxsRequest,
    IdxsResponse,
    LsEntry,
    LsRequest,
    LsResponse,
    MoveRequest,
    MoveResponse,
    RevertRequest,
    RevertResponse,
    RowsRequest,
    RowsResponse,
    StatusResponse,
)
from pixeltable import exceptions as excs

router = APIRouter()
_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@router.get('/pcli/v0/health', response_model=HealthResponse)
def pcli_health() -> HealthResponse:
    return HealthResponse(ok=True, pxt_version=pixeltable.__version__, pid=os.getpid(), started_at=_STARTED_AT)


@router.get('/api/pixeltable-health')
def dashboard_health() -> dict:
    # compat for the existing dashboard probe
    return {'status': 'ok'}


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------

# count() is SQL-bound (GIL-released during the query), so going wider than the DB can
# serve in parallel just queues. 16 is comfortably below pixeltable's default connection-pool ceiling.
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
    """Find children of `path` (slash-separated). Empty path = root."""
    parts = [p for p in path.split('/') if p]
    cur = tree
    for part in parts:
        for node in cur:
            if node['name'] == part and node['kind'] == 'directory':
                cur = node['entries']
                break
        else:
            raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f'Path {path!r} does not exist.')
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
            has_computed = any(c.get('is_computed') for c in cols.values())
            entry.flags = ('c' if has_computed else '') + ('i' if len(idxs) > 0 else '')
    return entry


def _fill_counts(entries: list[LsEntry]) -> None:
    # Resolve and count in the worker thread: pxt.get_table() returns a thread-local handle
    # that raises if reused across threads.
    targets = [e for e in entries if e.kind != 'dir']
    if len(targets) == 0:
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


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/describe', response_model=DescribeResponse)
def describe(req: DescribeRequest) -> DescribeResponse:
    t = pxt.get_table(req.path)
    return DescribeResponse(text=repr(t), metadata=jsonable_encoder(t.get_metadata()))


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/errors', response_model=ErrorsResponse)
def errors(req: ErrorsRequest) -> ErrorsResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()

    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise HTTPException(400, f'{req.path}: no primary key declared; pcli errors requires one')

    # errortype/errormsg are only addressable on stored computed columns; an unstored one
    # would raise UNSUPPORTED_OPERATION and fail the whole request.
    computed = [
        name
        for name, c in md['columns'].items()
        if c['is_computed'] and c.get('is_stored') and (req.col is None or name == req.col)
    ]
    if len(computed) == 0:
        return ErrorsResponse(entries=[])

    where = None
    for col in computed:
        cond = t[col].errortype != None  # pxt overrides != to return an Expr
        where = cond if where is None else where | cond

    select_args: list = [t[c] for c in pk_names]
    for col in computed:
        select_args += [t[col].errortype, t[col].errormsg]
    rows = t.where(where).select(*select_args).collect()

    entries: list[ErrorEntry] = []
    for r in rows:
        pk = {name: r[name] for name in pk_names}
        for col in computed:
            etype = r[f'{col}_errortype']
            emsg = r[f'{col}_errormsg']
            if etype is not None:
                entries.append(ErrorEntry(pk=pk, column=col, errortype=etype, errormsg=emsg))
    return ErrorsResponse(entries=entries)


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/history', response_model=HistoryResponse)
def history(req: HistoryRequest) -> HistoryResponse:
    versions = pxt.get_table(req.path).get_versions(req.n)
    return HistoryResponse(versions=jsonable_encoder(versions))


# ---------------------------------------------------------------------------
# columns / idxs (share _all_tables)
# ---------------------------------------------------------------------------


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


@router.post('/pcli/v0/idxs', response_model=IdxsResponse)
def idxs(req: IdxsRequest) -> IdxsResponse:
    paths = [req.path] if req.path else _all_tables()
    entries: list[IdxEntry] = []
    for path in paths:
        try:
            md = pxt.get_table(path).get_metadata()
        except excs.Error:
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


# ---------------------------------------------------------------------------
# rows / get (both default to stored columns to avoid surprising compute)
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/rows', response_model=RowsResponse)
def rows(req: RowsRequest) -> RowsResponse:
    if req.n <= 0:
        raise HTTPException(400, 'n must be > 0')
    t = pxt.get_table(req.path)
    cols_md = t.get_metadata()['columns']
    if req.cols is not None:
        missing = [c for c in req.cols if c not in cols_md]
        if len(missing) > 0:
            raise HTTPException(400, f'unknown columns: {",".join(missing)}')
        columns_list = list(req.cols)
    else:
        # Skip unstored computed columns by default: selecting one forces evaluation, which
        # can be slow or expensive (LLM calls, etc.). The user opts in via --cols.
        columns_list = [name for name, c in cols_md.items() if c.get('is_stored', True)]
    result = t.select(*[t[c] for c in columns_list]).limit(req.n).collect()
    out_rows: list[dict] = []
    for r in result:
        row: dict = {}
        for c in columns_list:
            v = r[c]
            if isinstance(v, PIL.Image.Image):
                row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
            else:
                row[c] = v
        out_rows.append(row)
    return RowsResponse(columns=columns_list, rows=jsonable_encoder(out_rows))


@router.post('/pcli/v0/get', response_model=GetResponse)
def get(req: GetRequest) -> GetResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()
    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise HTTPException(400, f'{req.path}: no primary key declared; pcli get requires one')
    if len(req.pk) != len(pk_names):
        raise HTTPException(400, f'{req.path}: expected {len(pk_names)} PK value(s) for {pk_names}, got {len(req.pk)}')

    cols_md = md['columns']
    if req.cols is not None:
        missing = [c for c in req.cols if c not in cols_md]
        if len(missing) > 0:
            raise HTTPException(400, f'unknown columns: {",".join(missing)}')
        cols = list(req.cols)
    else:
        # Skip unstored computed columns: a PK lookup shouldn't silently trigger arbitrary
        # computation (LLM calls, model inference, etc.). User opts in via --cols.
        cols = [name for name, c in cols_md.items() if c.get('is_stored', True)]

    where = None
    for name, val in zip(pk_names, req.pk):
        cond = t[name] == val
        where = cond if where is None else where & cond
    result = t.where(where).select(*[t[c] for c in cols]).limit(2).collect()
    if len(result) == 0:
        return GetResponse(pk_columns=pk_names, row=None)
    if len(result) > 1:
        raise HTTPException(
            500, f'{req.path}: {len(pk_names)}-column PK match returned multiple rows; catalog corruption?'
        )

    r = result[0]
    row: dict = {}
    for c in cols:
        v = r[c]
        if isinstance(v, PIL.Image.Image):
            row[c] = f'<Image {v.size[0]}x{v.size[1]} {v.mode}>'
        else:
            row[c] = v
    return GetResponse(pk_columns=pk_names, row=jsonable_encoder(row))


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/count', response_model=CountResponse)
def count(req: CountRequest) -> CountResponse:
    n = pxt.get_table(req.path).count()
    return CountResponse(path=req.path, count=n)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


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


def _redact_db_url(url: str | None) -> str | None:
    if url is None:
        return None
    try:
        return sa.make_url(url).render_as_string(hide_password=True)
    except Exception:
        return None


@router.get('/pcli/v0/status', response_model=StatusResponse)
def status(sizes: bool = False) -> StatusResponse:
    """Status snapshot. Pass `?sizes=1` to include media/file_cache disk usage (scans the directories)."""
    # Local import: pixeltable.dashboard.bridge pulls a heavy import chain that we don't
    # want on every daemon process start - only the status endpoint needs it.
    from pixeltable.dashboard import bridge

    s = bridge.get_status()
    cfg = s.get('config') or {}
    media_dir = cfg.get('media_dir')
    file_cache_dir = cfg.get('file_cache_dir')
    return StatusResponse(
        pxt_version=s['version'],
        pid=os.getpid(),
        started_at=_STARTED_AT,
        home=redact_home(cfg.get('home')),
        db_url=_redact_db_url(cfg.get('db_url')),
        media_dir=redact_home(media_dir),
        file_cache_dir=redact_home(file_cache_dir),
        media_size_bytes=_dir_size(media_dir) if sizes else None,
        file_cache_size_bytes=_dir_size(file_cache_dir) if sizes else None,
        total_tables=s['total_tables'],
        total_errors=s['total_errors'],
    )


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------

# Vars reported value-redacted. Match by suffix to catch provider-specific names
# (e.g. PIXELTABLE_DB_CONNECT_STR, plus any *_API_KEY / *_TOKEN / *_SECRET / *_PASSWORD).
_SENSITIVE_NAMES = {'PIXELTABLE_DB_CONNECT_STR'}
_SENSITIVE_SUFFIXES = ('_API_KEY', '_TOKEN', '_SECRET', '_PASSWORD')

# Common credential vars reported as presence-only (true/false) regardless of prefix.
_CREDENTIAL_VARS = (
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY',
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
    'MISTRAL_API_KEY',
    'GROQ_API_KEY',
    'COHERE_API_KEY',
    'TOGETHER_API_KEY',
    'HF_TOKEN',
    'HUGGINGFACE_API_KEY',
    'REPLICATE_API_TOKEN',
    'FIREWORKS_API_KEY',
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
)


def _is_sensitive(name: str) -> bool:
    return name in _SENSITIVE_NAMES or any(name.endswith(s) for s in _SENSITIVE_SUFFIXES)


def _redact_user_home(value: str) -> str:
    """Layer $HOME redaction on top of the shared $PIXELTABLE_HOME redaction.

    PIXELTABLE_HOME is typically nested under $HOME, so we run redact_home() first; a
    remaining absolute path inside $HOME (e.g. a custom file_cache_dir outside the pxt
    home) then becomes `$HOME/...`. Both sides go through realpath so symlinked layouts
    (macOS `/private/Users/me`) match.
    """
    after_pxt = redact_home(value) or value
    if after_pxt.startswith('$PIXELTABLE_HOME'):
        return after_pxt
    try:
        user_home = os.path.realpath(os.path.expanduser('~'))
        target = os.path.realpath(after_pxt)
    except OSError:
        return after_pxt
    if target == user_home:
        return '$HOME'
    if target.startswith(user_home + os.sep):
        return '$HOME' + target[len(user_home) :]
    return after_pxt


@router.get('/pcli/v0/env', response_model=EnvResponse)
def env() -> EnvResponse:
    reported_keys = [k for k in os.environ if k.startswith('PIXELTABLE_') or k == 'PCLI_PORT']
    env_vars: dict[str, str] = {}
    for k in reported_keys:
        raw = os.environ[k]
        env_vars[k] = '<redacted>' if _is_sensitive(k) else _redact_user_home(raw)
    credentials_present = {k: k in os.environ for k in _CREDENTIAL_VARS}
    config_file = os.environ.get('PIXELTABLE_CONFIG')
    return EnvResponse(
        env_vars=env_vars,
        config_file=_redact_user_home(config_file) if config_file is not None else None,
        credentials_present=credentials_present,
    )


# ---------------------------------------------------------------------------
# mutations: drop, move, revert
# ---------------------------------------------------------------------------


@router.post('/pcli/v0/drop', response_model=DropResponse)
def drop(req: DropRequest) -> DropResponse:
    if req.is_dir:
        pxt.drop_dir(req.path, force=req.cascade)
    else:
        pxt.drop_table(req.path, force=req.cascade)
    return DropResponse(path=req.path, dropped=True)


@router.post('/pcli/v0/move', response_model=MoveResponse)
def move(req: MoveRequest) -> MoveResponse:
    pxt.move(req.path, req.new_path)
    return MoveResponse(path=req.path, new_path=req.new_path)


@router.post('/pcli/v0/revert', response_model=RevertResponse)
def revert(req: RevertRequest) -> RevertResponse:
    if req.steps < 1:
        raise HTTPException(400, 'steps must be >= 1')
    t = pxt.get_table(req.path)
    from_version = t.get_metadata()['version']
    # Validate up front so the operation is all-or-nothing: without this, a multi-step
    # revert past version 0 would partially succeed before failing, leaving the table
    # at an intermediate version with no clean way for the caller to roll back.
    if req.steps > from_version:
        raise HTTPException(400, f'cannot revert {req.steps} step(s): {req.path} is at version {from_version}')
    for _ in range(req.steps):
        t.revert()
    to_version = pxt.get_table(req.path).get_metadata()['version']
    return RevertResponse(path=req.path, from_version=from_version, to_version=to_version)

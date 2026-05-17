import datetime
import os
from concurrent.futures import ThreadPoolExecutor

import PIL.Image
import sqlalchemy as sa
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

import pixeltable as pxt
from pcli import models
from pcli.paths import redact_home
from pixeltable import exceptions as excs
from pixeltable.types import TreeNode

router = APIRouter()
_STARTED_AT = datetime.datetime.now(datetime.timezone.utc).isoformat()


@router.get('/pcli/v0/health')
def pcli_health() -> models.HealthResponse:
    return models.HealthResponse(ok=True, pxt_version=pxt.__version__, pid=os.getpid(), started_at=_STARTED_AT)


@router.get('/api/pixeltable-health')
def dashboard_health() -> dict:
    # compat for the existing dashboard probe
    return {'status': 'ok'}


@router.post('/pcli/v0/ls')
def ls(req: models.LsRequest) -> models.LsResponse:
    tree = pxt.get_dir_tree()
    nodes = _collect_path_children(tree, req.path)
    if req.tree:
        return models.LsResponse(entries=[], tree={'path': req.path, 'entries': nodes})

    entries = [_to_entry(n, details=req.details) for n in nodes]
    if req.counts:
        _fill_counts(entries)
    return models.LsResponse(entries=entries)


@router.post('/pcli/v0/describe')
def describe(req: models.DescribeRequest) -> models.DescribeResponse:
    t = pxt.get_table(req.path)
    return models.DescribeResponse(text=repr(t), metadata=jsonable_encoder(t.get_metadata()))


@router.post('/pcli/v0/errors')
def errors(req: models.ErrorsRequest) -> models.ErrorsResponse:
    t = pxt.get_table(req.path)
    md = t.get_metadata()

    pk_names = md.get('primary_key')
    if pk_names is None or len(pk_names) == 0:
        raise HTTPException(400, f'{req.path}: no primary key declared; pcli errors requires one')

    # Validate req.col up front so a typo'd --col returns a 400 instead of a misleading
    # empty result; matches the unknown-column behavior of /rows and /get.
    if req.col is not None:
        col_md = md['columns'].get(req.col)
        if col_md is None:
            raise HTTPException(400, f'unknown column: {req.col}')
        if not (col_md['is_computed'] and col_md.get('is_stored')):
            raise HTTPException(400, f'{req.col}: not a stored computed column (pcli errors only addresses those)')

    # errortype/errormsg only apply to stored computed columns
    computed = [
        name
        for name, c in md['columns'].items()
        if c['is_computed'] and c.get('is_stored') and (req.col is None or name == req.col)
    ]
    if len(computed) == 0:
        return models.ErrorsResponse(entries=[])

    where = None
    for col in computed:
        cond = t[col].errortype != None  # pxt overrides != to return an Expr
        where = cond if where is None else where | cond

    select_args: list = [t[c] for c in pk_names]
    for col in computed:
        select_args += [t[col].errortype, t[col].errormsg]
    rows = t.where(where).select(*select_args).collect()

    entries: list[models.ErrorEntry] = []
    for r in rows:
        pk = {name: r[name] for name in pk_names}
        for col in computed:
            etype = r[f'{col}_errortype']
            emsg = r[f'{col}_errormsg']
            if etype is not None:
                entries.append(models.ErrorEntry(pk=pk, column=col, errortype=etype, errormsg=emsg))
    return models.ErrorsResponse(entries=entries)


@router.post('/pcli/v0/history')
def history(req: models.HistoryRequest) -> models.HistoryResponse:
    versions = pxt.get_table(req.path).get_versions(req.n)
    return models.HistoryResponse(versions=jsonable_encoder(versions))


@router.post('/pcli/v0/columns')
def columns(req: models.ColumnsRequest) -> models.ColumnsResponse:
    paths = [req.path] if req.path else _collect_table_paths()
    entries: list[models.ColumnEntry] = []
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
                models.ColumnEntry(
                    table=path,
                    column=name,
                    is_computed=c['is_computed'],
                    type_=c['type_'],
                    computed_with=c['computed_with'],
                    depends_on=c['depends_on'],
                )
            )
    return models.ColumnsResponse(entries=entries)


@router.post('/pcli/v0/idxs')
def idxs(req: models.IdxsRequest) -> models.IdxsResponse:
    paths = [req.path] if req.path else _collect_table_paths()
    entries: list[models.IdxEntry] = []
    for path in paths:
        try:
            md = pxt.get_table(path).get_metadata()
        except excs.Error:
            continue
        for name, idx in md['indices'].items():
            if req.embedding_only and idx['index_type'] != 'embedding':
                continue
            params = idx.get('parameters')
            entries.append(
                models.IdxEntry(
                    table=path,
                    name=name,
                    columns=idx['columns'],
                    index_type=idx['index_type'],
                    metric=params['metric'] if params is not None else None,
                    embedding=params['embedding'] if params is not None else None,
                )
            )
    return models.IdxsResponse(entries=entries)


@router.post('/pcli/v0/rows')
def rows(req: models.RowsRequest) -> models.RowsResponse:
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
        # skip unstored computed columns unless the user explicitly asked for them: evaluation can take
        # an unbounded amount of time
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
    return models.RowsResponse(columns=columns_list, rows=jsonable_encoder(out_rows))


@router.post('/pcli/v0/get')
def get(req: models.GetRequest) -> models.GetResponse:
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
        # skip unstored computed columns unless the user explicitly asked for them: evaluation can take
        # an unbounded amount of time
        cols = [name for name, c in cols_md.items() if c.get('is_stored', True)]

    where = None
    for name, val in zip(pk_names, req.pk):
        cond = t[name] == val
        where = cond if where is None else where & cond
    result = t.where(where).select(*[t[c] for c in cols]).limit(2).collect()
    if len(result) == 0:
        return models.GetResponse(pk_columns=pk_names, row=None)
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
    return models.GetResponse(pk_columns=pk_names, row=jsonable_encoder(row))


@router.post('/pcli/v0/count')
def count(req: models.CountRequest) -> models.CountResponse:
    n = pxt.get_table(req.path).count()
    return models.CountResponse(path=req.path, count=n)


@router.get('/pcli/v0/status')
def status(sizes: bool = False) -> models.StatusResponse:
    """Status snapshot. Pass sizes=1 to include media/file_cache disk usage (scans the directories)."""
    # Local import: pixeltable.dashboard.bridge pulls a heavy import chain that we don't
    # want on every daemon process start - only the status endpoint needs it.
    from pixeltable.dashboard import bridge

    s = bridge.get_status()
    cfg = s.get('config') or {}
    media_dir = cfg.get('media_dir')
    file_cache_dir = cfg.get('file_cache_dir')
    return models.StatusResponse(
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


@router.get('/pcli/v0/env')
def env() -> models.EnvResponse:
    reported_keys = [k for k in os.environ if k.startswith('PIXELTABLE_') or k == 'PCLI_PORT']
    env_vars: dict[str, str] = {}
    for k in reported_keys:
        raw = os.environ[k]
        env_vars[k] = '<redacted>' if _is_sensitive(k) else _redact_user_home(raw)
    credentials_present = {k: k in os.environ for k in _CREDENTIAL_VARS}
    config_file = os.environ.get('PIXELTABLE_CONFIG')
    return models.EnvResponse(
        env_vars=env_vars,
        config_file=_redact_user_home(config_file) if config_file is not None else None,
        credentials_present=credentials_present,
    )


@router.post('/pcli/v0/drop_table')
def drop_table(req: models.DropRequest) -> models.DropResponse:
    pxt.drop_table(req.path, force=req.cascade)
    return models.DropResponse(path=req.path, dropped=True)


@router.post('/pcli/v0/drop_dir')
def drop_dir(req: models.DropRequest) -> models.DropResponse:
    pxt.drop_dir(req.path, force=req.cascade)
    return models.DropResponse(path=req.path, dropped=True)


@router.post('/pcli/v0/move')
def move(req: models.MoveRequest) -> models.MoveResponse:
    pxt.move(req.path, req.new_path)
    return models.MoveResponse(path=req.path, new_path=req.new_path)


@router.post('/pcli/v0/revert')
def revert(req: models.RevertRequest) -> models.RevertResponse:
    if req.steps < 1:
        raise HTTPException(400, 'steps must be >= 1')
    t = pxt.get_table(req.path)
    from_version = t.get_metadata()['version']
    if req.steps > from_version:
        raise HTTPException(400, f'cannot revert {req.steps} step(s): {req.path} is at version {from_version}')
    for _ in range(req.steps):
        t.revert()
    to_version = pxt.get_table(req.path).get_metadata()['version']
    return models.RevertResponse(path=req.path, from_version=from_version, to_version=to_version)


# count() is SQL-bound (GIL-released during the query), so going wider than the DB can
# serve in parallel just queues. 16 is comfortably below pixeltable's default connection-pool ceiling.
_COUNT_POOL_WORKERS = 16


def _collect_path_children(tree: list[TreeNode], path: str) -> list[TreeNode]:
    """Find children of path (empty path = root)."""
    parts = path.split('/') if path else []
    cur = tree
    for part in parts:
        for node in cur:
            if node['name'] == part and node['kind'] == 'directory':
                cur = node['entries']
                break
        else:
            raise excs.NotFoundError(excs.ErrorCode.PATH_NOT_FOUND, f'Path {path!r} does not exist.')
    return cur


def _to_entry(node: TreeNode, details: bool) -> models.LsEntry:
    # pxt kinds: 'directory' | 'table' | 'view' | 'snapshot' | 'replica' (see pixeltable.types.TableKind)
    if node['kind'] == 'directory':
        return models.LsEntry(path=node['path'], kind='dir')
    entry = models.LsEntry(path=node['path'], kind=node['kind'], last_version=node['version'])
    if details:
        md = pxt.get_table(node['path']).get_metadata()
        cols = md.get('columns') or {}
        idxs = md.get('indices') or {}
        entry.num_cols = len(cols)
        has_computed = any(c.get('is_computed') for c in cols.values())
        entry.flags = ('c' if has_computed else '') + ('i' if len(idxs) > 0 else '')
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


def _redact_db_url(url: str | None) -> str | None:
    if url is None:
        return None
    try:
        return sa.make_url(url).render_as_string(hide_password=True)
    except Exception:
        return None


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

    PIXELTABLE_HOME is typically nested under $HOME, so we run redact_home() first; a remaining absolute path inside
    $HOME (e.g. a custom file_cache_dir outside the pxt home) then becomes $HOME/.... Both sides go through realpath
    so symlinked layouts (macOS /private/Users/me) match.
    """
    after_pxt = redact_home(value) or value
    if after_pxt.startswith('$PIXELTABLE_HOME'):
        return after_pxt
    # Only treat absolute paths and ~-prefixed paths as redactable. Anything else (bare scalars
    # like 'false'/'1', URLs like 's3://bucket', relative tokens) passes through untouched -
    # otherwise realpath() would expand them under $cwd and a wrong match to $HOME/... could
    # corrupt the reported value.
    if not (after_pxt.startswith('~') or os.path.isabs(after_pxt)):
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

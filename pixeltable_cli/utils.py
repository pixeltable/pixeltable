"""Primitives shared by the client and the in-process daemon: port, pidfile path, config-resolution
helpers, the identity fingerprint, and the hosted-CLI command helpers (URI parsing, printing, polling).
Stdlib-only so the client side can import without pulling in pxt or pydantic."""

import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_PORT = 22089

# Mirrors pixeltable.catalog.path._URI_RE (duplicated so this module stays stdlib-only): a hosted path is
# pxt://<org>:<db>/<in-catalog path>.
_PXT_URI_RE = re.compile(r'^pxt://(?P<org>[^:/]+)(?::(?P<db>[^/]+))?(?:/(?P<rest>.*))?$')


def _resolve_pixeltable_home() -> str:
    """Mirror of pixeltable.config.Config's home-directory resolution"""
    return str(Path(os.environ.get('PIXELTABLE_HOME') or '~/.pixeltable').expanduser().resolve())


def _resolve_pixeltable_pgdata(home: str) -> str:
    """Mirrors env.py: PIXELTABLE_PGDATA if set, else {home}/pgdata."""
    return str(Path(os.environ.get('PIXELTABLE_PGDATA') or f'{home}/pgdata').expanduser().resolve())


def _resolve_pixeltable_config_file(home: str) -> str:
    """Mirrors config.py: PIXELTABLE_CONFIG if set, else {home}/config.toml."""
    return str(Path(os.environ.get('PIXELTABLE_CONFIG') or f'{home}/config.toml').expanduser().resolve())


def get_port() -> int:
    raw = os.environ.get('PXT_PORT')
    if raw is None or raw == '':
        return DEFAULT_PORT
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f'PXT_PORT must be an integer port; got {raw!r}') from None


def pidfile_path() -> str:
    """Per-port pidfile path. The port parameterization isolates daemons running on
    different ports so they don't read or stomp each other's PID."""
    return os.path.join(_resolve_pixeltable_home(), f'pxt-daemon-{get_port()}.pid')


def validate_path_shape(path: str) -> str | None:
    """Return an error message if path violates pxt path shape rules, else None. Empty is allowed.

    A hosted URI pxt://<org>:<db>/<in-catalog path> is accepted; only its in-catalog portion is shape-checked
    here. The org/db and the overall URI form are validated by pixeltable when the path is resolved.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in path):
        return f'pxt paths must not contain control characters; got {path!r}'
    if path.startswith('pxt://'):
        m = _PXT_URI_RE.match(path)
        if m is None:
            return f'invalid URI; expected pxt://<org>:<db>/<path>, got {path!r}'
        in_catalog = m.group('rest') or ''
    else:
        in_catalog = path
        if in_catalog.startswith('/'):
            return f"pxt paths are relative; drop the leading '/' (use '' for root). Got {path!r}"
    if '.' in in_catalog:
        return f"pxt paths use '/' as the separator; got {path!r}"
    if in_catalog.endswith('/'):
        return f"pxt paths must not end with '/'; got {path!r}"
    if '//' in in_catalog:
        return f"pxt paths must not contain empty components ('//'); got {path!r}"
    return None


# Identity fingerprint keys
_IDENTITY_KEYS: tuple[str, ...] = (
    'pxt_version',
    'pxt_install_dir',
    'python_executable',
    'pixeltable_home',
    'pixeltable_pgdata',
    'pixeltable_config_file',
    'pixeltable_env',
)


def _pxt_version() -> str | None:
    try:
        return importlib.metadata.version('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None


def _pxt_install_dir() -> str | None:
    """Stdlib-only equivalent of Path(pixeltable.__file__).parent. Returns None if pixeltable isn't installed."""
    try:
        dist = importlib.metadata.distribution('pixeltable')
    except importlib.metadata.PackageNotFoundError:
        return None
    return str(Path(str(dist.locate_file('pixeltable'))).resolve())


# Env-var keys whose values may contain credentials and must not be returned verbatim
_SENSITIVE_NAME_PARTS: tuple[str, ...] = ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PASSWD', 'CONNECT_STR')


def _is_sensitive_env_name(name: str) -> bool:
    upper = name.upper()
    return any(part in upper for part in _SENSITIVE_NAME_PARTS)


def _redact_env_value(name: str, value: str) -> str:
    if not _is_sensitive_env_name(name):
        return value
    digest = hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]
    return f'sha256:{digest}'


def _snapshot_pixeltable_env(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Returns dict mapping PIXELTABLE_* env vars to their redacted values."""
    env = os.environ if environ is None else environ
    return {k: _redact_env_value(k, env[k]) for k in sorted(env) if k.startswith('PIXELTABLE_')}


def identity() -> dict[str, Any]:
    pxt_version = _pxt_version()
    pxt_install_dir = _pxt_install_dir()
    # Surfacing this here turns a broken pixeltable install into one clear error instead
    # of a daemon that 500s on every /health call and respawns in a tight loop.
    if pxt_version is None or pxt_install_dir is None:
        raise RuntimeError(
            "pixeltable package metadata not found (importlib.metadata can't locate the "
            "'pixeltable' distribution). Reinstall with: pip install --force-reinstall pixeltable"
        )
    home = _resolve_pixeltable_home()
    return {
        'pxt_version': pxt_version,
        'pxt_install_dir': pxt_install_dir,
        'python_executable': sys.executable,
        'pixeltable_home': home,
        'pixeltable_pgdata': _resolve_pixeltable_pgdata(home),
        'pixeltable_config_file': _resolve_pixeltable_config_file(home),
        'pixeltable_env': _snapshot_pixeltable_env(),
    }


_DB_POLL_INTERVAL = 5
_DB_POLL_TIMEOUT = 600
_SVC_POLL_INTERVAL = 5
_SVC_POLL_TIMEOUT = 300
_RUNTIME_POLL_INTERVAL = 10
_RUNTIME_POLL_TIMEOUT = 900


def _split_pxt_uri(uri: str) -> tuple[str | None, str | None, str | None]:
    """Return (org, db, path) from a pxt:// URI, or (None, None, None) on parse failure."""
    m = _PXT_URI_RE.match(uri)
    if m is None:
        return None, None, None
    return m.group('org') or None, m.group('db') or None, m.group('rest') or None


def parse_db_uri(uri: str, prog: str = 'pxt') -> tuple[str, str]:
    """Parse pxt://org:db and return (org_slug, db_slug). Exits on error."""
    org, db, path = _split_pxt_uri(uri)
    if not org or not db or path is not None:
        print(f'{prog}: error: URI must be pxt://org:db, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return org, db


def parse_org_uri(uri: str, prog: str = 'pxt') -> str:
    """Parse pxt://org and return org_slug. Exits on error."""
    org, db, path = _split_pxt_uri(uri)
    if not org or db is not None or path is not None:
        print(f'{prog}: error: URI must be pxt://org, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return org


def parse_base_uri(uri: str, prog: str = 'pxt') -> tuple[str, str, str]:
    """Parse pxt://org:db[/<path>] and return (org, db, base_path). Exits on error."""
    org, db, path = _split_pxt_uri(uri)
    if not org or not db:
        print(f'{prog}: error: --base-uri must be pxt://org:db[/<dir>], got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return org, db, path or ''


def parse_service_uri(uri: str, prog: str = 'pxt') -> tuple[str, str, str]:
    """Parse pxt://org:db/services/<name> and return (org, db, svc_name). Exits on error."""
    org, db, path = _split_pxt_uri(uri)
    if not org or not db or not path or not path.startswith('services/'):
        print(f'{prog}: error: URI must be pxt://org:db/services/<name>, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    svc_name = path[len('services/') :]
    if not svc_name or '/' in svc_name:
        print(
            f'{prog}: error: URI must be pxt://org:db/services/<name> with no extra path, got {uri!r}', file=sys.stderr
        )
        sys.exit(2)
    return org, db, svc_name


def _fmt_age(age_s: int) -> str:
    if age_s < 60:
        return f'{age_s}s'
    if age_s < 3600:
        return f'{age_s // 60}m'
    if age_s < 86400:
        h = age_s // 3600
        m = (age_s % 3600) // 60
        return f'{h}h{m}m' if m else f'{h}h'
    d = age_s // 86400
    h = (age_s % 86400) // 3600
    return f'{d}d{h}h' if h else f'{d}d'


def _print_workers(workers: list[dict[str, Any]]) -> None:
    if not workers:
        return
    print(f'  {"POD ID":<52}{"STATUS":<22}{"READY":>5}  {"RESTARTS":>8}  AGE')
    for w in workers:
        pod_id = w.get('pod_id', '')
        status = w.get('status', '')
        ready = f'{w.get("ready", 0)}/{w.get("total", 0)}'
        restarts = str(w.get('restarts', 0))
        age = _fmt_age(w.get('age_s', 0))
        print(f'  {pod_id:<52}{status:<22}{ready:>5}  {restarts:>8}  {age}')


def print_db(db: dict[str, Any]) -> None:
    name = db.get('db_name') or db.get('db_slug', '')
    state = db.get('state', '')
    location = db.get('location', '')
    region = db.get('region', '')
    endpoint = db.get('endpoint') or ''
    print(f'{name}  state={state}  {location}/{region}  {endpoint}'.rstrip())
    _print_workers(db.get('workers') or [])


def print_service(svc: dict[str, Any]) -> None:
    name = svc.get('service_name', '')
    state = svc.get('state', '')
    base = svc.get('base_path', '')
    workers_max = svc.get('workers_max')
    if workers_max is not None:
        workers_str = f'workers={svc.get("workers_min", 1)}-{workers_max}'
    else:
        workers_str = f'workers={svc.get("workers_min", 1)}'
    endpoint = svc.get('endpoint') or ''
    print(f'{name}  state={state}  base={base}  {workers_str}  {endpoint}'.rstrip())
    # Print route URLs from service_config
    svc_config_str = svc.get('service_config')
    if svc_config_str and endpoint:
        try:
            svc_cfg = json.loads(svc_config_str) if isinstance(svc_config_str, str) else svc_config_str
            prefix = svc_cfg.get('prefix', '')
            for route in svc_cfg.get('routes', []):
                method = route.get('method', 'POST').upper()
                path = route.get('path', '')
                print(f'  {method}  {endpoint}{prefix}{path}')
        except Exception:
            pass
    _print_workers(svc.get('workers') or [])


def print_org(org: dict[str, Any]) -> None:
    slug = org.get('org_slug', '')
    org_id = org.get('org_id', '')
    default_db = org.get('default_db_slug') or ''
    line = f'{slug}  id={org_id}'
    if default_db:
        line += f'  default_db={default_db}'
    print(line)


def poll_db(org_slug: str, db_slug: str, pending_states: frozenset[str], label: str) -> dict[str, Any]:
    """Poll daemon GET .../dbs/{db_slug} until state leaves pending_states."""
    # imported lazily: rich is heavy, and importing http at module scope would create a utils<->http cycle
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from pixeltable_cli.client.http import get

    db: dict[str, Any] = {}
    deadline = time.monotonic() + _DB_POLL_TIMEOUT
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        TimeElapsedColumn(),
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        progress.add_task(label, total=None)
        while time.monotonic() < deadline:
            time.sleep(_DB_POLL_INTERVAL)
            try:
                resp = get(f'/api/orgs/{org_slug}/dbs/{db_slug}')
                db = resp.get('database', resp) if isinstance(resp, dict) else {}
            except SystemExit:
                raise
            except Exception:
                pass
            if db.get('state') not in pending_states:
                break
    return db


def poll_svc(org_slug: str, db_slug: str, svc_name: str, pending_states: frozenset[str], label: str) -> dict[str, Any]:
    """Poll daemon GET .../services/{svc_name} until state leaves pending_states."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from pixeltable_cli.client.http import get

    svc: dict[str, Any] = {}
    deadline = time.monotonic() + _SVC_POLL_TIMEOUT
    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        TimeElapsedColumn(),
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        progress.add_task(label, total=None)
        while time.monotonic() < deadline:
            time.sleep(_SVC_POLL_INTERVAL)
            try:
                resp = get(f'/api/orgs/{org_slug}/dbs/{db_slug}/services/{svc_name}')
                svc = resp.get('service', resp) if isinstance(resp, dict) else {}
            except SystemExit:
                raise
            except Exception:
                pass
            if svc.get('state') not in pending_states:
                break
    return svc

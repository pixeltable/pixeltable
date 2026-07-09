"""Helpers for cloud CLI commands: URI parsing, printing, and polling via daemon.

These run in the CLI process without loading pixeltable.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from .http import get

_DB_POLL_INTERVAL = 5
_DB_POLL_TIMEOUT = 600
_SVC_POLL_INTERVAL = 5
_SVC_POLL_TIMEOUT = 300
_RUNTIME_POLL_INTERVAL = 10
_RUNTIME_POLL_TIMEOUT = 900


def parse_db_uri(uri: str, prog: str = 'pxt') -> tuple[str, str]:
    """Parse pxt://org:db and return (org_slug, db_slug). Exits on error."""
    org, db, _ = _split_pxt_uri(uri)
    if not org or not db:
        print(f'{prog}: error: URI must be pxt://org:db, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return org, db


def parse_org_uri(uri: str, prog: str = 'pxt') -> str:
    """Parse pxt://org and return org_slug. Exits on error."""
    org, _, _ = _split_pxt_uri(uri)
    if not org:
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
    if not svc_name:
        print(f'{prog}: error: URI must include a service name', file=sys.stderr)
        sys.exit(2)
    return org, db, svc_name


def _split_pxt_uri(uri: str) -> tuple[str | None, str | None, str | None]:
    """Return (org, db, path) from a pxt:// URI. Returns (None, None, None) on parse failure."""
    if not uri.startswith('pxt://'):
        return None, None, None
    rest = uri[6:]
    if ':' in rest:
        colon = rest.index(':')
        org = rest[:colon]
        after_colon = rest[colon + 1 :]
        if '/' in after_colon:
            slash = after_colon.index('/')
            db = after_colon[:slash]
            path = after_colon[slash + 1 :] or None
        else:
            db = after_colon
            path = None
    else:
        org = rest.split('/')[0] if '/' in rest else rest
        db = None
        path = None
    return org or None, db or None, path


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
                resp = get(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}')
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
                resp = get(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services/{svc_name}')
                svc = resp.get('service', resp) if isinstance(resp, dict) else {}
            except SystemExit:
                raise
            except Exception:
                pass
            if svc.get('state') not in pending_states:
                break
    return svc


def poll_db_runtime(org_slug: str, db_slug: str) -> dict[str, Any]:
    """Poll until DB leaves UPDATING state (runtime build)."""
    db: dict[str, Any] = {}
    deadline = time.monotonic() + _RUNTIME_POLL_TIMEOUT
    print('Waiting for runtime build', end='', flush=True)
    while time.monotonic() < deadline:
        time.sleep(_RUNTIME_POLL_INTERVAL)
        try:
            resp = get(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}')
            db = resp.get('database', resp) if isinstance(resp, dict) else {}
        except SystemExit:
            break
        except Exception:
            pass
        print('.', end='', flush=True)
        if db.get('state') != 'UPDATING':
            break
    print()
    return db

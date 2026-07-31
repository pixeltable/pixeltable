"""URI parsing, output formatting and state polling shared by the hosted-CLI commands
(`pxt db`, `pxt service`, `pxt org`)."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections.abc import Iterator
from typing import Any

from pixeltable_cli.utils import split_pxt_uri

from .utils import get_request, print_aligned

DB_POLL_INTERVAL = 5
DB_POLL_TIMEOUT = 600
SVC_POLL_INTERVAL = 5
SVC_POLL_TIMEOUT = 300
RUNTIME_POLL_INTERVAL = 10
RUNTIME_POLL_TIMEOUT = 900


def parse_db_uri(uri: str, prog: str = 'pxt') -> tuple[str, str]:
    """Parse pxt://org:db and return (org, db). Exits on error."""
    parts = split_pxt_uri(uri)
    if parts is None or parts.db is None or parts.path is not None:
        print(f'{prog}: error: URI must be pxt://org:db, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return parts.org, parts.db


def resolve_db_uri(db_uri: str | None, prog: str = 'pxt') -> tuple[str, str]:
    """Parse pxt://org:db and return (org, db), defaulting to the configured pixeltable.db_uri. Exits on error."""
    if db_uri is None:
        resp = get_request('/api/config')
        entries = resp.get('entries', []) if isinstance(resp, dict) else []
        configured = next(
            (e.get('value') for e in entries if e.get('section') == 'pixeltable' and e.get('key') == 'db_uri'), None
        )
        if configured is None:
            print(
                f'{prog}: error: no database URI given, and no db_uri is set in the Pixeltable config file',
                file=sys.stderr,
            )
            sys.exit(2)
        db_uri = configured
    return parse_db_uri(db_uri, prog=prog)


def parse_org_uri(uri: str, prog: str = 'pxt') -> str:
    """Parse pxt://org and return org. Exits on error."""
    parts = split_pxt_uri(uri)
    if parts is None or parts.db is not None or parts.path is not None:
        print(f'{prog}: error: URI must be pxt://org, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return parts.org


def parse_base_uri(uri: str, prog: str = 'pxt') -> tuple[str, str, str]:
    """Parse pxt://org:db[/<path>] and return (org, db, base_path). Exits on error."""
    parts = split_pxt_uri(uri)
    if parts is None or parts.db is None:
        print(f'{prog}: error: --base-uri must be pxt://org:db[/<dir>], got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return parts.org, parts.db, parts.path or ''


def parse_service_uri(uri: str, prog: str = 'pxt') -> tuple[str, str, str]:
    """Parse pxt://org:db/services/<name> and return (org, db, svc_name). Exits on error."""
    parts = split_pxt_uri(uri)
    if parts is None or parts.db is None or parts.path is None or not parts.path.startswith('services/'):
        print(f'{prog}: error: URI must be pxt://org:db/services/<name>, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    svc_name = parts.path[len('services/') :]
    if svc_name == '' or '/' in svc_name:
        print(
            f'{prog}: error: URI must be pxt://org:db/services/<name> with no extra path, got {uri!r}', file=sys.stderr
        )
        sys.exit(2)
    return parts.org, parts.db, svc_name


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
    rows = [
        [
            w.get('pod_id', ''),
            w.get('status', ''),
            f'{w.get("ready", 0)}/{w.get("total", 0)}',
            str(w.get('restarts', 0)),
            _fmt_age(w.get('age_s', 0)),
        ]
        for w in workers
    ]
    print_aligned(['POD ID', 'STATUS', 'READY', 'RESTARTS', 'AGE'], rows, right_align={2, 3}, indent='  ')


def print_db(db: dict[str, Any]) -> None:
    name = db.get('db_name') or db.get('db', '')
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
    name = org.get('org', '')
    org_id = org.get('org_id', '')
    default_db = org.get('default_db') or ''
    line = f'{name}  id={org_id}'
    if default_db:
        line += f'  default_db={default_db}'
    print(line)


@contextlib.contextmanager
def _spinner(label: str | None) -> Iterator[None]:
    """Display a transient progress spinner showing label for the duration of the block; None displays nothing."""
    if label is None:
        yield
        return

    # imported lazily: rich is a heavy import, and a poll without a label never reaches this
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    with Progress(
        SpinnerColumn(),
        TextColumn('[progress.description]{task.description}'),
        TimeElapsedColumn(),
        transient=True,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        progress.add_task(label, total=None)
        yield


def poll_state(
    endpoint: str,
    params: dict[str, str],
    result_key: str,
    pending_states: set[str],
    interval: float,
    timeout: float,
    label: str | None,
) -> dict[str, Any]:
    """Poll a daemon route until the resource's 'state' leaves pending_states, or timeout seconds elapse.

    Returns the last response read, or an empty dict if none succeeded. A failed read is retried until the
    deadline, so a resource that is briefly unreachable mid-transition doesn't abort the wait.
    """
    result: dict[str, Any] = {}
    deadline = time.monotonic() + timeout
    with _spinner(label):
        while time.monotonic() < deadline:
            time.sleep(interval)
            try:
                resp = get_request(endpoint, params)
            except SystemExit:
                raise
            except Exception:
                continue
            result = resp.get(result_key, resp) if isinstance(resp, dict) else {}
            if result.get('state') not in pending_states:
                break
    return result


def exit_unless_reached(result: dict[str, Any], expected_state: str, operation: str) -> None:
    """Exit with 1 unless the operation reached expected_state."""
    state = result.get('state')
    if state == expected_state:
        return
    seen = 'no state was read' if state is None else f'last state: {state}'
    print(f'pxt: {operation} did not complete ({seen})', file=sys.stderr)
    sys.exit(1)


def poll_db(org: str, db: str, pending_states: set[str], label: str | None) -> dict[str, Any]:
    """Poll a hosted database until its state leaves pending_states."""
    return poll_state(
        '/api/db', {'org': org, 'db': db}, 'database', pending_states, DB_POLL_INTERVAL, DB_POLL_TIMEOUT, label
    )


def poll_svc(org: str, db: str, svc_name: str, pending_states: set[str], label: str | None) -> dict[str, Any]:
    """Poll a hosted service until its state leaves pending_states."""
    return poll_state(
        '/api/service',
        {'org': org, 'db': db, 'service_name': svc_name},
        'service',
        pending_states,
        SVC_POLL_INTERVAL,
        SVC_POLL_TIMEOUT,
        label,
    )

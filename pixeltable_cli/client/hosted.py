"""URI parsing, output formatting and state polling shared by the hosted-CLI commands
(`pxt db`, `pxt service`, `pxt org`)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from typing import Any, Iterator

from pixeltable_cli import models
from pixeltable_cli.types import DbState
from pixeltable_cli.utils import split_pxt_uri

from .utils import get_request, print_aligned

DB_POLL_INTERVAL = 5

_TRANSITIONAL = frozenset(s.value for s in DbState if s.is_transitional)
DB_POLL_TIMEOUT = 600


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
        resp = models.ConfigResponse.model_validate(get_request('/api/config'))
        configured = next((e.value for e in resp.entries if (e.section, e.key) == ('pixeltable', 'db_uri')), None)
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


def add_logs_args(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by `pxt db logs` and `pxt service logs`."""
    parser.add_argument('--since', default='1h', help='how far back to read: 30s, 10m, 1h, 2d (default: 1h)')
    parser.add_argument(
        '--tail',
        type=int,
        default=200,
        dest='limit',
        help='the newest N lines in the window, at most 10000 (default: 200)',
    )
    parser.add_argument(
        '--include-health', action='store_true', dest='include_health', help='keep the GET /health probe lines'
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')


def print_logs(target: dict[str, str], args: argparse.Namespace) -> None:
    """Print the log of target: {'org', 'db'} names a database's pod, {'service'} names a service."""
    params = {**target, 'since': args.since, 'limit': args.limit, 'include_health': args.include_health}
    resp = get_request('/api/logs', params)
    records = resp.get('records', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(records))
        return
    if len(records) == 0:
        print(f'No log records in the last {args.since}.')
        return
    for r in records:
        print(r['line'])


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


def _fmt_project(resources: dict[str, Any]) -> str | None:
    """Describe a database's project by its file count and archive digest."""
    fp = resources.get('fingerprint')
    if fp is None:
        return None
    files = fp.get('files') or {}
    # must match ProjectFingerprint.archive_digest() in pixeltable/utils/project.py;
    # inlined to avoid importing from pixeltable here, which adds substantial load latency
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8]
    md_version = resources.get('pxt_md_version') or 0
    return f'{len(files)} files  archive {digest}  md_version {md_version}'


_RESOURCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ('cpu', 'cpu', ''),
    ('memory', 'memory_mb', ' MiB'),
    ('disk', 'disk_gb', ' GiB'),
    ('workers', 'workers', ''),
)


def print_db(report: dict[str, Any], workers: list[dict[str, Any]] | None = None) -> None:
    """Print one database's report: its state, its resources, and what an update would change."""
    current = report.get('current')
    if current is None:
        print(f'{report.get("db", "")}  absent')
        return
    print(f'{report.get("db", "")}  {current.get("state", "")}')

    running = current.get('resources') or {}
    target = report.get('target_resources') or {}
    rows: list[tuple[str, str]] = []
    for label, field, unit in _RESOURCE_ROWS:
        now, want = running.get(field), target.get(field)
        if now is None and want is None:
            continue
        text = f'{now}{unit}' if now is not None else '-'
        if want is not None and want != now:
            text += f' -> {want}{unit}   pending'
        rows.append((label, text))

    now_project, want_project = _fmt_project(running), _fmt_project(target)
    if now_project is not None or want_project is not None:
        text = now_project or '-'
        if want_project is not None and want_project != now_project:
            text += f' -> {want_project}   pending'
        rows.append(('project', text))
    if running.get('default_bucket') is not None:
        rows.append(('bucket', running['default_bucket']))

    outcome = current.get('last_build_outcome')
    if outcome is not None and outcome != 'SUCCEEDED':
        rows.append(('build', outcome))
    for label, field in (('error', 'last_build_error'), ('reason', 'failure_reason')):
        if current.get(field) is not None:
            rows.append((label, str(current[field])))

    width = max((len(label) for label, _ in rows), default=0)
    for label, text in rows:
        print(f'  {label.ljust(width)}  {text}')
    _print_workers(workers or [])


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
    pending = ' (update pending)' if svc.get('update_pending') is True else ''
    print(f'{name}  state={state}{pending}  base={base}  {workers_str}  {endpoint}'.rstrip())
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
def spinner(label: str | None) -> Iterator[None]:
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


def db_state(response: dict[str, Any]) -> str | None:
    return ((response.get('report') or {}).get('current') or {}).get('state')


def exit_unless_reached(response: dict[str, Any], expected_state: DbState, operation: str) -> None:
    """Exit with 1 unless the database reached expected_state."""
    state = db_state(response)
    if state == expected_state:
        return
    seen = 'no state was read' if state is None else f'last state: {state}'
    print(f'pxt: {operation} did not complete ({seen})', file=sys.stderr)
    sys.exit(1)


def poll_db(org: str, db: str, label: str | None) -> dict[str, Any]:
    """Poll a hosted database until nothing is in flight, and return the whole response.

    Returns an empty dict if no read succeeded. A failed read is retried until the deadline, so a
    database briefly unreachable mid-transition does not abort the wait.
    """
    response: dict[str, Any] = {}
    deadline = time.monotonic() + DB_POLL_TIMEOUT
    with spinner(label):
        while time.monotonic() < deadline:
            time.sleep(DB_POLL_INTERVAL)
            try:
                resp = get_request('/api/db', {'org': org, 'db': db})
            except SystemExit:
                raise
            except Exception:
                continue
            response = resp if isinstance(resp, dict) else {}
            state = db_state(response)
            # an unrecognized state is one a newer control plane reports, and waiting on it would hang
            if state is None or state not in _TRANSITIONAL:
                break
    return response

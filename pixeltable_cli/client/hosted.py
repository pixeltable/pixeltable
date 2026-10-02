"""URI parsing and output formatting shared by the hosted-CLI commands (`pxt db`, `pxt service`, `pxt org`)."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, NoReturn

from pixeltable_cli import models
from pixeltable_cli.types import GenerationReceipt
from pixeltable_cli.utils import split_pxt_uri

from .utils import EXIT_CHANGES_PENDING, EXIT_ERROR, get_request, post_request, print_aligned

RECEIPT_POLL_INTERVAL = 5.0
RECEIPT_WAIT_TIMEOUT = 3 * 3600.0
EXIT_INTERRUPTED = 130

_ENDPOINT_TIMEOUT = 60.0
_ENDPOINT_PROBE_TIMEOUT = 10.0


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
    return f'{len(files)} files  archive {digest}'


_RESOURCE_ROWS: tuple[tuple[str, str, str], ...] = (
    ('cpu', 'cpu', ''),
    ('memory', 'memory_mb', ' MiB'),
    ('disk', 'disk_gb', ' GiB'),
    ('workers', 'workers', ''),
)


def print_db(report: dict[str, Any], workers: list[dict[str, Any]] | None = None) -> None:
    """Print one database's report: what it serves, what it was last asked for where that differs, and its pods.

    Whether a change is still under way is the current generation's phase, not a comparison made here.
    """
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
            text += f'   desired {want}{unit}'
        rows.append((label, text))

    now_project, want_project = _fmt_project(running), _fmt_project(target)
    if now_project is not None or want_project is not None:
        text = now_project or '-'
        if want_project is not None and want_project != now_project:
            text += f'   desired {want_project}'
        rows.append(('project', text))
    if current.get('md_version') is not None:
        rows.append(('md_version', str(current['md_version'])))
    if running.get('default_bucket') is not None:
        rows.append(('bucket', running['default_bucket']))

    outcome = current.get('last_build_outcome')
    if outcome is not None and outcome != 'SUCCEEDED':
        rows.append(('build', outcome))
    receipt = report.get('receipt')
    if receipt is not None:
        rows.append(('generation', describe_receipt(GenerationReceipt.model_validate(receipt))))
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
def spinner(label: str | None) -> Iterator[Callable[[str], None]]:
    """Display a transient progress spinner showing label for the duration of the block; None displays nothing.

    Yields a function that replaces the label.
    """
    if label is None:
        yield lambda _: None
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
        task = progress.add_task(label, total=None)
        yield lambda text: progress.update(task, description=text)


def describe_receipt(receipt: GenerationReceipt) -> str:
    """One line saying where a generation stands."""
    if receipt.observed:
        state = 'took effect'
    elif receipt.superseded:
        state = 'replaced by a newer one'
    elif receipt.failed:
        state = f'FAILED: {receipt.failure_reason}'
    elif receipt.error is not None:
        state = f'retrying after: {receipt.error.message}'
    else:
        state = receipt.progress
    return f'{receipt.generation}  {state}'


def print_receipt(receipt: GenerationReceipt) -> None:
    """Print where a generation stands, and its warnings."""
    print(f'{receipt.resource} generation {describe_receipt(receipt)}')
    for warning in receipt.warnings:
        print(f'  warning: {warning}')


def await_receipts(
    db_uri: str, receipts: list[GenerationReceipt], *, show: bool, status_command: str
) -> list[GenerationReceipt]:
    """Poll receipts until each settles, showing where the first unsettled one stands, and return them.

    Waiting only reads: Ctrl-C, or a change still unsettled after RECEIPT_WAIT_TIMEOUT, stops the wait and exits,
    and Pixeltable Cloud carries the change out regardless. The timeout exits EXIT_CHANGES_PENDING.
    """
    if len(receipts) == 0:
        print('pxt: Pixeltable Cloud accepted the change without a receipt to wait on', file=sys.stderr)
        sys.exit(EXIT_ERROR)
    current = receipts
    deadline = time.monotonic() + RECEIPT_WAIT_TIMEOUT
    try:
        with spinner(_waiting_label(current) if show else None) as set_label:
            while not all(r.settled for r in current) and not any(r.unsuccessful for r in current):
                if time.monotonic() >= deadline:
                    break
                set_label(_waiting_label(current))
                time.sleep(RECEIPT_POLL_INTERVAL)
                resp = post_request(
                    '/api/receipts', {'db_uri': db_uri, 'receipts': [r.model_dump(mode='json') for r in current]}
                )
                current = models.ReceiptsResponse.model_validate(resp).receipts
    except KeyboardInterrupt:
        _exit_stopped_waiting(status_command)
    if not all(r.settled for r in current) and not any(r.unsuccessful for r in current):
        print(
            f'pxt: still in progress after {RECEIPT_WAIT_TIMEOUT / 3600:.0f}h: {_waiting_label(current).rstrip(" .")}\n'
            f'Pixeltable Cloud carries the change out regardless; `{status_command}` shows its progress.',
            file=sys.stderr,
        )
        sys.exit(EXIT_CHANGES_PENDING)
    return current


def exit_unless_observed(receipts: list[GenerationReceipt], *, retry_command: str) -> None:
    """Exit with EXIT_ERROR, saying why, unless every receipt took effect."""
    for receipt in receipts:
        if receipt.failed:
            print(
                f'pxt: {receipt.resource} generation {receipt.generation} failed: {receipt.failure_reason}\n'
                f'Run `{retry_command}` to try it again, or change it and update again.',
                file=sys.stderr,
            )
            sys.exit(EXIT_ERROR)
        if receipt.superseded:
            print(
                f'pxt: {receipt.resource} generation {receipt.generation} was replaced by a newer update before it '
                'took effect',
                file=sys.stderr,
            )
            sys.exit(EXIT_ERROR)
        if receipt.outcome is not None and not receipt.observed:
            print(
                f'pxt: {receipt.resource} generation {receipt.generation} ended as {receipt.outcome}, which this '
                'version of Pixeltable does not know; upgrade it to read the outcome',
                file=sys.stderr,
            )
            sys.exit(EXIT_ERROR)


def await_endpoint(endpoint: str, *, status_command: str) -> None:
    """Poll a hosted service's endpoint until a request reaches its pod rather than the gateway.

    A service takes effect when its pod is ready, which can be moments before the gateway routes to it; until then
    the gateway answers 502. Any status the pod itself produced, 404 included, means the route is through. Ctrl-C
    stops the wait as it does in await_receipts().
    """
    deadline = time.monotonic() + _ENDPOINT_TIMEOUT
    try:
        while True:
            try:
                with urllib.request.urlopen(f'{endpoint}/health', timeout=_ENDPOINT_PROBE_TIMEOUT):
                    return
            except urllib.error.HTTPError as e:
                if e.code not in (502, 503, 504):
                    return
            except (urllib.error.URLError, TimeoutError):
                pass
            if time.monotonic() >= deadline:
                print(f'pxt: {endpoint} did not answer within {_ENDPOINT_TIMEOUT:.0f}s', file=sys.stderr)
                sys.exit(EXIT_ERROR)
            time.sleep(RECEIPT_POLL_INTERVAL)
    except KeyboardInterrupt:
        _exit_stopped_waiting(status_command)


def _exit_stopped_waiting(status_command: str) -> NoReturn:
    print(
        f'\npxt: stopped waiting; Pixeltable Cloud carries the change out regardless. '
        f'`{status_command}` shows its progress.',
        file=sys.stderr,
    )
    sys.exit(EXIT_INTERRUPTED)


def _waiting_label(receipts: list[GenerationReceipt]) -> str:
    pending = next((r for r in receipts if not r.settled), receipts[0])
    return f'{pending.resource} generation {describe_receipt(pending)} ...'

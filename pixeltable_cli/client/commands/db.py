"""`pxt db {create,list,status,start,stop,update,update-runtime,delete} <uri>` - manage hosted databases."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import IO, Any

from ..hosted import (
    RUNTIME_POLL_INTERVAL,
    RUNTIME_POLL_TIMEOUT,
    Spinner,
    exit_unless_reached,
    parse_org_uri,
    poll_db,
    poll_state,
    print_db,
    resolve_db_uri,
)
from ..parser import Parser
from ..utils import get_request, post_request


def _elapsed(seconds: float) -> str:
    if seconds < 1:
        return 'less than a second'
    whole = int(round(seconds))
    return f'{whole}s' if whole < 60 else f'{whole // 60}m{whole % 60:02d}s'


def _print_stages(stages: list[tuple[str, str, float, str]]) -> None:
    """What is still on screen once the transient progress displays have cleared.

    Not transient, so the timings survive the run; one display for every row, because splitting the
    failed one onto another stream reorders it against its neighbours. The exit status carries failure.
    """
    # imported lazily to match the rest of the client: rich is a heavy import for a stdlib-only CLI
    from rich.progress import Column, Progress, TextColumn

    with Progress(
        TextColumn('[progress.description]{task.description}', table_column=Column(min_width=8)),
        TextColumn('{task.fields[state]}', table_column=Column(min_width=10)),
        TextColumn('{task.fields[took]}', table_column=Column(min_width=18)),
        # fold rather than clip: the reason a stage failed is the one thing that must stay readable
        TextColumn('{task.fields[detail]}', table_column=Column(overflow='fold')),
        transient=False,
        redirect_stdout=False,
        redirect_stderr=False,
    ) as progress:
        for name, state, seconds, detail in stages:
            progress.add_task(name, total=None, state=state, took=_elapsed(seconds), detail=detail)

EPILOG = """\
Examples:
  pxt db create pxt://org:db
  pxt db list pxt://org
  pxt db status pxt://org:db
  pxt db start pxt://org:db
  pxt db stop pxt://org:db
  pxt db update pxt://org:db --workers 2
  pxt db update-runtime pxt://org:db --project-dir .
  pxt db delete pxt://org:db
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db', description='manage hosted databases', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('create', help='create a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--location', default='aws', help='Cloud provider (default: aws)')
    p.add_argument('--region', default='us-east-1', help='Region (default: us-east-1)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('list', help='list hosted databases for an org')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('start', help='start (wake) a stopped hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('stop', help='stop (sleep) a running hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('update', help='update worker count or resource limits')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--workers', type=int, default=None, help='Number of proxy daemon workers')
    p.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    p.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    p.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('update-runtime', help='rebuild the Python runtime for a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument(
        '--project-dir',
        default=None,
        metavar='DIR',
        help='Project directory containing pyproject.toml and uv.lock (default: current directory)',
    )
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'create':
        _create(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)
    elif args.action == 'start':
        _start(args)
    elif args.action == 'stop':
        _stop(args)
    elif args.action == 'update':
        _update(args)
    elif args.action == 'update-runtime':
        _update_runtime(args)
    elif args.action == 'delete':
        _delete(args)


def _create(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db create')
    resp = post_request('/api/dbs', {'org': org, 'db': db, 'location': args.location, 'region': args.region})
    result = resp.get('database', resp) if isinstance(resp, dict) else {}
    if result.get('state') == 'PROVISIONING':
        result = poll_db(org, db, {'PROVISIONING'}, f"Database '{db}' is provisioning...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'AVAILABLE', f'creating database {db!r}')


def _list(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt db list')
    resp = get_request('/api/dbs', {'org': org})
    dbs = resp.get('databases', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(dbs))
    elif not dbs:
        print('No databases.')
    else:
        for db in dbs:
            print_db(db)


def _status(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db status')
    resp = get_request('/api/db', {'org': org, 'db': db})
    result = resp.get('database', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)


def _start(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db start')
    post_request('/api/db/start', {'org': org, 'db': db})
    result = poll_db(org, db, {'UPDATING', 'STARTING'}, f"Database '{db}' is starting...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'AVAILABLE', f'starting database {db!r}')


def _stop(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db stop')
    post_request('/api/db/stop', {'org': org, 'db': db})
    result = poll_db(org, db, {'STOPPING'}, f"Database '{db}' is stopping...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'STOPPED', f'stopping database {db!r}')


def _update(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db update')
    resp = post_request(
        '/api/db/update',
        {
            'org': org,
            'db': db,
            'workers': args.workers,
            'cpu': args.cpu,
            'memory_mb': args.memory_mb,
            'disk_gb': args.disk_gb,
        },
    )
    result = resp.get('database', resp) if isinstance(resp, dict) else {}
    if result.get('state') == 'UPDATING':
        result = poll_db(org, db, {'UPDATING'}, f"Database '{db}' is updating...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'AVAILABLE', f'updating database {db!r}')


def _delete(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db delete')
    post_request('/api/db/delete', {'org': org, 'db': db})
    if args.json_output:
        print(json.dumps({'deleted': db}))
    else:
        print(f"Deleted database '{db}'.")


class _ProgressReader:
    """File-like wrapper that advances a progress bar as the wrapped file is read."""

    f: IO[bytes]
    bar: Any

    def __init__(self, f: IO[bytes], bar: Any) -> None:
        self._f = f
        self._bar = bar

    def read(self, size: int = -1) -> bytes:
        data = self._f.read(size)
        self._bar.update(len(data))
        return data


def _update_runtime(args: argparse.Namespace) -> None:
    # imported lazily: these pull in pixeltable and tqdm, which the stdlib-only client avoids
    # loading for the other db subcommands
    from tqdm import tqdm

    from pixeltable.serving.deploy import build_db_runtime_bundle

    org, db = resolve_db_uri(args.db_uri, prog='pxt db update-runtime')

    project_dir = (Path(args.project_dir) if args.project_dir is not None else Path.cwd()).resolve()

    # Markers that identify a project directory: a project file or a supported lockfile.
    # Keep this list in sync with deploy.py.
    required = ('pyproject.toml', 'uv.lock', 'poetry.lock', 'requirements.txt', 'pixeltable.toml')
    if not any((project_dir / f).exists() for f in required):
        print(
            f'pxt: error: no project files (pyproject.toml, uv.lock, poetry.lock, requirements.txt, '
            f'or pixeltable.toml) found in {project_dir}.\n'
            'Run from your project directory or pass --project-dir.',
            file=sys.stderr,
        )
        sys.exit(1)

    show_progress = not args.json_output
    stages: list[tuple[str, str, float, str]] = []  # name, state, seconds, detail

    def finish(code: int) -> None:
        if not args.json_output:
            _print_stages(stages)
        sys.exit(code)

    started = time.monotonic()
    try:
        bundle_path = build_db_runtime_bundle(project_dir, show_progress=show_progress)
    except Exception as exc:
        stages.append(('bundle', 'failed', time.monotonic() - started, str(exc)))
        finish(1)
    stages.append(
        ('bundle', 'succeeded', time.monotonic() - started, f'{bundle_path.stat().st_size / (1024 * 1024):.1f} MB')
    )

    started = time.monotonic()
    try:
        url_resp = get_request('/api/db/upload-url', {'org': org, 'db': db})
        presigned_url = url_resp['presigned_url']
        bundle_s3_key = url_resp['bundle_s3_key']

        bundle_size = bundle_path.stat().st_size
        with (
            bundle_path.open('rb') as f,
            tqdm(
                desc='Uploading runtime bundle', total=bundle_size, unit='B', unit_scale=True, disable=not show_progress
            ) as bar,
        ):
            # urllib streams a file-like body in chunks, which lets the bar advance during the upload
            req = urllib.request.Request(presigned_url, data=_ProgressReader(f, bar), method='PUT')
            req.add_header('Content-Type', 'application/octet-stream')
            req.add_header('Content-Length', str(bundle_size))
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status >= 400:
                    raise RuntimeError(f'Bundle upload failed: HTTP {r.status}')
    except Exception as exc:
        stages.append(('upload', 'failed', time.monotonic() - started, str(exc)))
        bundle_path.unlink(missing_ok=True)
        finish(1)
    finally:
        bundle_path.unlink(missing_ok=True)
    stages.append(('upload', 'succeeded', time.monotonic() - started, ''))

    post_request('/api/db/update-runtime', {'org': org, 'db': db, 'bundle_s3_key': bundle_s3_key})

    # The server moves through its own stages; watching each reading is the only way to time them
    # separately, since the record only ever holds the stage currently in flight.
    watching = {'stage': '', 'since': time.monotonic()}

    def on_poll(res: dict[str, Any], spinner: Spinner) -> None:
        stage = str((res.get('update_runtime_status') or {}).get('stage', '')).lower()
        if not stage or stage == watching['stage']:
            return
        now = time.monotonic()
        if watching['stage']:
            stages.append((str(watching['stage']), 'succeeded', now - float(watching['since']), ''))
        watching['stage'], watching['since'] = stage, now
        spinner.label(f'{stage} ...')

    label = None if args.json_output else 'build ...'
    started = time.monotonic()
    result = poll_state(
        '/api/db',
        {'org': org, 'db': db},
        'database',
        {'UPDATING'},
        RUNTIME_POLL_INTERVAL,
        RUNTIME_POLL_TIMEOUT,
        label,
        on_poll=on_poll,
    )
    # A server that reports no stages still ran them; time the wait as one span rather than inventing names.
    server_stage = str(watching['stage']) or 'build+deploy'
    server_seconds = time.monotonic() - float(watching['since'] if watching['stage'] else started)

    # The server reports the stage the update reached and what became of it: "build failed" and
    # "deploy failed" are different problems and only the stage separates them. A server that does not
    # report the pair still says whether the build worked, which is enough to name the outcome.
    status = result.get('update_runtime_status') or {}
    stage = str(status.get('stage', '')).lower()
    state = str(status.get('state', '')).lower()
    last_build_state = result.get('last_build_state')
    build_failed = state == 'failed' or last_build_state == 'FAILED'
    succeeded = state == 'succeeded' or (not state and last_build_state == 'ACTIVE')
    build_error = status.get('error') or result.get('last_build_error') or ''
    if build_failed:
        stages.append((server_stage, 'failed', server_seconds, build_error))
    elif succeeded:
        stages.append((server_stage, 'succeeded', server_seconds, ''))
    else:
        stages.append((server_stage, str(result.get('state', 'unknown')).lower(), server_seconds, ''))

    if args.json_output:
        print(json.dumps(result))
    else:
        _print_stages(stages)
    if build_failed:
        sys.exit(1)
    exit_unless_reached(result, 'AVAILABLE', f'the runtime build of database {db!r}')

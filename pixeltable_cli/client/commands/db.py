"""`pxt db {create,list,status,start,stop,update,update-runtime,delete} <uri>` - manage hosted databases."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import IO, Any

from ..hosted import (
    RUNTIME_POLL_INTERVAL,
    RUNTIME_POLL_TIMEOUT,
    exit_unless_reached,
    parse_org_uri,
    poll_db,
    poll_state,
    print_db,
    resolve_db_uri,
)
from ..parser import Parser
from ..utils import get_request, post_request

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

    from pixeltable.service.build_context import build_context

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
    context_path = build_context(project_dir, show_progress=show_progress, db_name=f'pxt://{org}:{db}')

    try:
        url_resp = get_request('/api/db/upload-url', {'org': org, 'db': db})
        presigned_url = url_resp['presigned_url']
        build_context_key = url_resp['build_context_key']

        context_size = context_path.stat().st_size
        with (
            context_path.open('rb') as f,
            tqdm(
                desc='Uploading project', total=context_size, unit='B', unit_scale=True, disable=not show_progress
            ) as bar,
        ):
            # urllib streams a file-like body in chunks, which lets the bar advance during the upload
            req = urllib.request.Request(presigned_url, data=_ProgressReader(f, bar), method='PUT')
            req.add_header('Content-Type', 'application/octet-stream')
            req.add_header('Content-Length', str(context_size))
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status >= 400:
                    raise RuntimeError(f'Bundle upload failed: HTTP {r.status}')
    finally:
        context_path.unlink(missing_ok=True)

    post_request('/api/db/update-runtime', {'org': org, 'db': db, 'build_context_key': build_context_key})

    label = None if args.json_output else 'Waiting for runtime build (this may take 10 minutes or longer) ...'
    result = poll_state(
        '/api/db', {'org': org, 'db': db}, 'database', {'UPDATING'}, RUNTIME_POLL_INTERVAL, RUNTIME_POLL_TIMEOUT, label
    )

    build_failed = result.get('last_build_state') == 'FAILED'
    build_error = result.get('last_build_error') or ''
    if not args.json_output:
        final_state = result.get('state', '')
        if build_failed:
            print(f'Runtime build failed: {build_error}', file=sys.stderr)
        elif final_state:
            print(f'Runtime build {final_state.lower()}.')

    if args.json_output:
        print(json.dumps(result))
    if build_failed:
        sys.exit(1)
    exit_unless_reached(result, 'AVAILABLE', f'the runtime build of database {db!r}')

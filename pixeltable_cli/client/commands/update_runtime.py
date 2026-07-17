from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from pixeltable.serving.deploy import build_db_runtime_bundle
from pixeltable_cli.utils import _RUNTIME_POLL_INTERVAL, _RUNTIME_POLL_TIMEOUT, parse_db_uri

from ..http import get, post
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db update-runtime', description='Rebuild the Python runtime for a hosted database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument(
        '--project-dir',
        default=None,
        metavar='DIR',
        help='Project directory containing pyproject.toml and uv.lock (default: current directory)',
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt db update-runtime')

        if args.project_dir is not None:
            os.chdir(args.project_dir)

        project_dir = Path.cwd().resolve()

        required = ('pyproject.toml', 'uv.lock', 'pixeltable.toml')
        if not any((project_dir / f).exists() for f in required):
            print(
                f'pxt: error: no pyproject.toml, uv.lock, or pixeltable.toml found in {project_dir}.\n'
                'Run from your project directory or pass --project-dir.',
                file=sys.stderr,
            )
            sys.exit(1)

        if not args.json_output:
            print('Building runtime bundle...', end=' ', flush=True)
        bundle_path = build_db_runtime_bundle(project_dir)
        if not args.json_output:
            size_mb = bundle_path.stat().st_size / (1024 * 1024)
            print(f'done ({size_mb:.1f} MB)')

        try:
            if not args.json_output:
                print('Uploading bundle...', end=' ', flush=True)
            url_resp = get(f'/api/orgs/{org_slug}/dbs/{db_slug}/upload-url')
            presigned_url = url_resp['presigned_url']
            bundle_s3_key = url_resp['bundle_s3_key']

            with bundle_path.open('rb') as fh:
                req = urllib.request.Request(presigned_url, data=fh, method='PUT')
                req.add_header('Content-Type', 'application/octet-stream')
                req.add_header('Content-Length', str(bundle_path.stat().st_size))
                with urllib.request.urlopen(req, timeout=300) as r:
                    if r.status >= 400:
                        raise RuntimeError(f'Bundle upload failed: HTTP {r.status}')
            if not args.json_output:
                print('done')
        finally:
            bundle_path.unlink(missing_ok=True)

        post(f'/api/orgs/{org_slug}/dbs/{db_slug}/update-runtime', {'bundle_s3_key': bundle_s3_key})

        # Poll until state leaves UPDATING
        db: dict = {}
        deadline = time.monotonic() + _RUNTIME_POLL_TIMEOUT
        if not args.json_output:
            print('Waiting for runtime build', end='', flush=True)
        while time.monotonic() < deadline:
            time.sleep(_RUNTIME_POLL_INTERVAL)
            try:
                resp = get(f'/api/orgs/{org_slug}/dbs/{db_slug}')
                db = resp.get('database', resp) if isinstance(resp, dict) else {}
            except SystemExit:
                break
            except Exception:
                pass
            if not args.json_output:
                print('.', end='', flush=True)
            if db.get('state') != 'UPDATING':
                break
        build_failed = db.get('last_build_state') == 'FAILED'
        build_error = db.get('last_build_error') or ''
        if not args.json_output:
            print()
            final_state = db.get('state', '')
            if build_failed:
                print(f'Runtime build failed: {build_error}', file=sys.stderr)
            elif final_state:
                print(f'Runtime build {final_state.lower()}.')
            else:
                print('Timed out waiting for runtime build.')

        if args.json_output:
            print(json.dumps(db))

        if build_failed:
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

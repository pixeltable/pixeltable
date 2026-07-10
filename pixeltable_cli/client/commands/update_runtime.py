from __future__ import annotations

import json
import sys
import time
import urllib.request

from ..parser import Parser

_RUNTIME_POLL_INTERVAL = 10
_RUNTIME_POLL_TIMEOUT = 900


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db update-runtime', description='Rebuild the Python runtime for a cloud-hosted database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--config', default=None, metavar='FILE', help='Path to an additional config file (TOML)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable import config as pxt_config
    from pixeltable.serving.deploy import build_runtime_bundle

    from ..cloud import parse_db_uri
    from ..http import get, post

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt db update-runtime')

        if args.config is not None:
            pxt_config.Config.init({}, additional_config_files=[args.config])

        if not args.json_output:
            print('Building runtime bundle...', end=' ', flush=True)
        bundle_path = build_runtime_bundle()
        if not args.json_output:
            size_mb = bundle_path.stat().st_size / (1024 * 1024)
            print(f'done ({size_mb:.1f} MB)')

        try:
            if not args.json_output:
                print('Uploading bundle...', end=' ', flush=True)
            url_resp = get(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/upload-url')
            presigned_url = url_resp['presigned_url']
            bundle_s3_key = url_resp['bundle_s3_key']

            with bundle_path.open('rb') as fh:
                bundle_data = fh.read()
            req = urllib.request.Request(presigned_url, data=bundle_data, method='PUT')
            req.add_header('Content-Type', 'application/octet-stream')
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status >= 400:
                    raise RuntimeError(f'Bundle upload failed: HTTP {r.status}')
            if not args.json_output:
                print('done')
        finally:
            bundle_path.unlink(missing_ok=True)

        post(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/update-runtime', {'bundle_s3_key': bundle_s3_key})

        # Poll until state leaves UPDATING
        db: dict = {}
        deadline = time.monotonic() + _RUNTIME_POLL_TIMEOUT
        if not args.json_output:
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
            if not args.json_output:
                print('.', end='', flush=True)
            if db.get('state') != 'UPDATING':
                break
        if not args.json_output:
            print()
            final_state = db.get('state', '')
            if final_state:
                print(f'Runtime build {final_state.lower()}.')
            else:
                print('Timed out waiting for runtime build.')

        if args.json_output:
            print(json.dumps(db))
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

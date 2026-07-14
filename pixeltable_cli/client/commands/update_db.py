from __future__ import annotations

import json
import sys

from ..cloud import parse_db_uri, poll_db, print_db
from ..http import post
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt db update', description='Update worker count or resource limits for a cloud-hosted database.'
    )
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--workers', type=int, default=None, help='Number of proxy daemon workers')
    parser.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    parser.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    parser.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt db update')
        resp = post(
            f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/update',
            {'workers': args.workers, 'cpu': args.cpu, 'memory_mb': args.memory_mb, 'disk_gb': args.disk_gb},
        )
        db = resp.get('database', resp) if isinstance(resp, dict) else {}
        if db.get('state') == 'UPDATING':
            db = poll_db(org_slug, db_slug, frozenset({'UPDATING'}), f"Database '{db_slug}' is updating...")
        if args.json_output:
            print(json.dumps(db))
        else:
            print_db(db)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

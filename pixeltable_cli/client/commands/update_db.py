from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt db update',
        description='Update worker count or resource limits for a cloud-hosted database.',
    )
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--workers', type=int, default=None, help='Number of proxy daemon workers')
    parser.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    parser.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    parser.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.share.deploy_client import database_update

    try:
        p = PxtUri(args.db_uri)
        if p.db is None:
            parser.error('db_uri must be pxt://org:db')

        database_update(
            p.org,
            p.db,
            workers=args.workers,
            cpu=args.cpu,
            memory_mb=args.memory_mb,
            disk_gb=args.disk_gb,
            json_output=args.json_output,
        )
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

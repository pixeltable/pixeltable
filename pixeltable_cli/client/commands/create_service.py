from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service create', description='Create a service from a table in a cloud-hosted database.')
    parser.add_argument('table_uri', help='Table URI: pxt://org:db/tables/<path>')
    parser.add_argument(
        '--name', required=True, help='Service name (must match a [[service]] block in pixeltable.toml)'
    )
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (default: 1)')
    parser.add_argument('--cpu', type=float, default=0.5, help='CPU cores per worker (default: 0.5)')
    parser.add_argument(
        '--memory', type=int, default=512, dest='memory_mb', help='Memory per worker in MB (default: 512)'
    )
    parser.add_argument('--disk', type=int, default=10, dest='disk_gb', help='Disk per worker in GB (default: 10)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_table_uri, poll_svc, print_service, read_service_config
    from ..http import post

    try:
        org_slug, db_slug, table_path = parse_table_uri(args.table_uri, prog='pxt service create')

        service_config = read_service_config(args.name)

        resp = post(
            f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services',
            {
                'service_name': args.name,
                'table_path': table_path,
                'workers': args.workers,
                'cpu': args.cpu,
                'memory_mb': args.memory_mb,
                'disk_gb': args.disk_gb,
                'service_config': service_config,
            },
        )
        svc = resp.get('service', resp) if isinstance(resp, dict) else {}
        if svc.get('state') in ('DEPLOYING', 'STARTING'):
            svc = poll_svc(
                org_slug,
                db_slug,
                args.name,
                frozenset({'DEPLOYING', 'STARTING'}),
                f"Service '{args.name}' is deploying...",
            )
        if args.json_output:
            print(json.dumps(svc))
        else:
            print_service(svc)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

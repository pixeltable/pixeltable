from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt service update',
        description='Update a service. Reads route config from [[service]] in pixeltable.toml.',
    )
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--workers', type=int, default=None, help='New minimum worker count')
    parser.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    parser.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    parser.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_service_uri, print_service, read_service_config
    from ..http import post

    try:
        org_slug, db_slug, svc_name = parse_service_uri(args.service_uri, prog='pxt service update')

        service_config = read_service_config(svc_name)

        resp = post(
            f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services/{svc_name}/update',
            {
                'workers': args.workers,
                'cpu': args.cpu,
                'memory_mb': args.memory_mb,
                'disk_gb': args.disk_gb,
                'service_config': service_config,
            },
        )
        svc = resp.get('service', resp) if isinstance(resp, dict) else {}
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

from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service create', description='Create a cloud-hosted service.')
    parser.add_argument('name', help='Service name (must match a [[pixeltable.service]] block in the config)')
    parser.add_argument(
        '--base-uri',
        required=True,
        dest='base_uri',
        metavar='URI',
        help='pxt://org:db[/<dir>] — database and base path prefix for resolving relative table paths in routes',
    )
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (default: 1)')
    parser.add_argument('--cpu', type=float, default=0.5, help='CPU cores per worker (default: 0.5)')
    parser.add_argument(
        '--memory', type=int, default=512, dest='memory_mb', help='Memory per worker in MB (default: 512)'
    )
    parser.add_argument('--disk', type=int, default=10, dest='disk_gb', help='Disk per worker in GB (default: 10)')
    parser.add_argument('--config', default=None, metavar='FILE', help='Path to an additional config file (TOML)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable import config as pxt_config
    from pixeltable.serving._config import lookup_service_config

    from ..cloud import parse_base_uri, poll_svc, print_service
    from ..http import post

    try:
        org_slug, db_slug, base_path = parse_base_uri(args.base_uri, prog='pxt service create')

        if args.config is not None:
            pxt_config.Config.init({}, additional_config_files=[args.config])
        service_config = lookup_service_config(args.name).model_dump_json()

        resp = post(
            f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services',
            {
                'service_name': args.name,
                'base_path': base_path,
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

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
    parser.add_argument('--memory', type=int, default=512, dest='memory_mb', help='Memory per worker in MB (default: 512)')
    parser.add_argument('--disk', type=int, default=10, dest='disk_gb', help='Disk per worker in GB (default: 10)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.serving._config import lookup_service_config
    from pixeltable.share.deploy_client import service_create

    try:
        p = PxtUri(args.table_uri)
        if p.db is None or not p.path or not p.path.startswith('tables/'):
            parser.error('table_uri must be pxt://org:db/tables/<path>')
        table_path = p.path[len('tables/') :]
        if not table_path:
            parser.error('table_uri must include a table path, e.g. pxt://org:db/tables/my_table')

        service_cfg = lookup_service_config(args.name)
        service_config_json = service_cfg.model_dump_json()

        service_create(
            p.org,
            p.db,
            args.name,
            table_path,
            workers_min=args.workers,
            workers_max=args.workers,
            cpu=args.cpu,
            memory_mb=args.memory_mb,
            disk_gb=args.disk_gb,
            service_config=service_config_json,
            json_output=args.json_output,
        )
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

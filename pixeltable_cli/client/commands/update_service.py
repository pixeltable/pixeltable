from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt service update',
        description='Update a service (routes and/or worker count). '
        'Route changes re-read the [[service]] block from pixeltable.toml.',
    )
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--workers', type=int, default=None, help='New minimum worker count')
    parser.add_argument(
        '--routes',
        action='store_true',
        dest='update_routes',
        help='Re-read [[service]] from pixeltable.toml and push updated route config',
    )
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.serving._config import lookup_service_config
    from pixeltable.share.deploy_client import service_update

    try:
        p = PxtUri(args.service_uri)
        if p.db is None or not p.path or not p.path.startswith('services/'):
            parser.error('service_uri must be pxt://org:db/services/<name>')
        service_name = p.path[len('services/') :]
        if not service_name:
            parser.error('service_uri must include a service name')

        service_config_json = None
        if args.update_routes:
            service_cfg = lookup_service_config(service_name)
            service_config_json = service_cfg.model_dump_json()

        service_update(
            p.org,
            p.db,
            service_name,
            workers_min=args.workers,
            service_config=service_config_json,
            json_output=args.json_output,
        )
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

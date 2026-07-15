from __future__ import annotations

import json
import sys

from pixeltable import config as pxt_config, exceptions as excs
from pixeltable.serving._config import lookup_service_config

from ..cloud import parse_service_uri, poll_svc, print_service
from ..http import post
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(
        prog='pxt service update',
        description='Update a service. Reads route config from [[pixeltable.service]] in the Pixeltable config.',
    )
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--workers', type=int, default=None, help='New minimum worker count')
    parser.add_argument('--cpu', type=float, default=None, help='CPU cores per worker')
    parser.add_argument('--memory', type=int, default=None, dest='memory_mb', help='Memory per worker in MB')
    parser.add_argument('--disk', type=int, default=None, dest='disk_gb', help='Disk per worker in GB')
    parser.add_argument('--config', default=None, metavar='FILE', help='Path to an additional config file (TOML)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug, db_slug, svc_name = parse_service_uri(args.service_uri, prog='pxt service update')

        additional_files = [args.config] if args.config is not None else []
        pxt_config.Config.init({}, additional_config_files=additional_files)
        try:
            service_config = lookup_service_config(svc_name).model_dump_json()
        except excs.NotFoundError:
            service_config = None

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
        if svc.get('state') == 'UPDATING':
            svc = poll_svc(org_slug, db_slug, svc_name, frozenset({'UPDATING'}), f"Service '{svc_name}' is updating...")
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

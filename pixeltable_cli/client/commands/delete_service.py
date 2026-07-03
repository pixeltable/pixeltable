from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service delete', description='Delete a service from a cloud-hosted database.')
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.catalog.path import Path as PxtPath
    from pixeltable.share.deploy_client import service_delete

    try:
        p = PxtPath.parse(args.service_uri)
        if p.org is None or p.db is None:
            parser.error('service_uri must be pxt://org:db/services/<name>')
        comps = list(p.components)
        if comps and comps[0] == 'services':
            comps = comps[1:]
        svc_name = '.'.join(comps)
        if not svc_name:
            parser.error('service_uri must include a service name')
        service_delete(p.org, p.db, svc_name, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

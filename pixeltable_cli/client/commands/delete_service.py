from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service delete', description='Delete a service from a cloud-hosted database.')
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.share.deploy_client import service_delete

    try:
        p = PxtUri(args.service_uri)
        svc_name = p.service
        if p.db is None or svc_name is None:
            parser.error('service_uri must be pxt://org:db/services/<name>')
        service_delete(p.org, p.db, svc_name, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

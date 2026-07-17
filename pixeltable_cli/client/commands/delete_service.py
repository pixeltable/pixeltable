from __future__ import annotations

import json
import sys

from pixeltable_cli.utils import parse_service_uri

from ..http import post
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service delete', description='Delete a service from a hosted database.')
    parser.add_argument('service_uri', help='Service URI: pxt://org:db/services/<name>')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug, db_slug, svc_name = parse_service_uri(args.service_uri, prog='pxt service delete')
        post(f'/api/orgs/{org_slug}/dbs/{db_slug}/services/{svc_name}/delete', {})
        if args.json_output:
            print(json.dumps({'deleted': svc_name}))
        else:
            print(f"Deleted service '{svc_name}'.")
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

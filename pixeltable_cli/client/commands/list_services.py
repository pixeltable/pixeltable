from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt service list', description='List services in a cloud-hosted database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_db_uri, print_service
    from ..http import get

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt service list')
        resp = get(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/services')
        svcs = resp.get('services', []) if isinstance(resp, dict) else []
        if args.json_output:
            print(json.dumps(svcs))
        elif not svcs:
            print(f"No services in database '{db_slug}'.")
        else:
            for svc in svcs:
                print_service(svc)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

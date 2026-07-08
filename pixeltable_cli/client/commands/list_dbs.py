from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db list', description='List cloud-hosted databases for an org.')
    parser.add_argument('org_uri', help='Org URI: pxt://org')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_org_uri, print_db
    from ..http import get

    try:
        org_slug = parse_org_uri(args.org_uri, prog='pxt db list')
        resp = get(f'/api/cloud/orgs/{org_slug}/dbs')
        dbs = resp.get('databases', []) if isinstance(resp, dict) else []
        if args.json_output:
            print(json.dumps(dbs))
        elif not dbs:
            print('No databases.')
        else:
            for db in dbs:
                print_db(db)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

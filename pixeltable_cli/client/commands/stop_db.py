from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db stop', description='Stop (sleep) a running cloud-hosted Pixeltable database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from ..cloud import parse_db_uri
    from ..http import post

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt db stop')
        resp = post(f'/api/cloud/orgs/{org_slug}/dbs/{db_slug}/stop', {})
        if args.json_output:
            print(json.dumps(resp))
        else:
            print(resp.get('message', 'stopped') if isinstance(resp, dict) else 'stopped')
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

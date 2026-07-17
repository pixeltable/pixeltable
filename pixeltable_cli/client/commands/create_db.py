from __future__ import annotations

import json
import sys

from pixeltable_cli.utils import parse_db_uri, poll_db, print_db

from ..http import post
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db create', description='Create a hosted Pixeltable database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--location', default='aws', help='Cloud provider (default: aws)')
    parser.add_argument('--region', default='us-east-1', help='Region (default: us-east-1)')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug, db_slug = parse_db_uri(args.db_uri, prog='pxt db create')
        resp = post(f'/api/orgs/{org_slug}/dbs', {'db_slug': db_slug, 'location': args.location, 'region': args.region})
        db = resp.get('database', resp) if isinstance(resp, dict) else {}
        if db.get('state') == 'PROVISIONING':
            db = poll_db(org_slug, db_slug, frozenset({'PROVISIONING'}), f"Database '{db_slug}' is provisioning...")
        if args.json_output:
            print(json.dumps(db))
        else:
            print_db(db)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

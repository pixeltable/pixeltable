from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db status', description='Show status of a cloud-hosted database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.share.deploy_client import database_get

    try:
        p = PxtUri(args.db_uri)
        if p.db is None:
            parser.error('db_uri must be pxt://org:db')
        database_get(p.org, p.db, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

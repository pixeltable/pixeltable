from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db stop', description='Stop (sleep) a running cloud-hosted Pixeltable database.')
    parser.add_argument('db_uri', help='Database URI: pxt://org:db or just the db slug')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.share.deploy_client import database_stop

    try:
        uri = args.db_uri
        if uri.startswith('pxt://'):
            from pixeltable.catalog.path import Path as PxtPath

            p = PxtPath.parse(uri, allow_empty_path=True)
            if p.db is None:
                parser.error('db_uri must be pxt://org:db')
            database_stop(p.db, org_slug=p.org, json_output=args.json_output)
        else:
            database_stop(uri, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db list', description='List cloud-hosted databases for an org.')
    parser.add_argument('org_uri', help='Org URI: pxt://org')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.catalog.path import Path as PxtPath
    from pixeltable.share.deploy_client import database_list

    try:
        p = PxtPath.parse(args.org_uri, allow_empty_path=True)
        if p.org is None:
            parser.error('org_uri must be pxt://org')
        database_list(p.org, json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

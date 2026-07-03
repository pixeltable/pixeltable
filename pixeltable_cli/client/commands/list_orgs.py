from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org list', description='List all organizations accessible to the current API key.')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.share.deploy_client import org_list

    try:
        org_list(json_output=args.json_output)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

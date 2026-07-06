from __future__ import annotations

import json
import sys

from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org status', description='Show status of an organization.')
    parser.add_argument('org_uri', help='Org URI: pxt://org')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    from pixeltable.service.utils import PxtUri
    from pixeltable.share.deploy_client import org_get

    try:
        p = PxtUri(args.org_uri)
        result = org_get(p.org, json_output=args.json_output)
        if result is None:
            sys.exit(1)
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

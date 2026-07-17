from __future__ import annotations

import json
import sys

from pixeltable_cli.utils import print_org

from ..http import get
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org list', description='List all organizations accessible to the current API key.')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        resp = get('/api/orgs')
        orgs = resp.get('orgs', []) if isinstance(resp, dict) else []
        if args.json_output:
            print(json.dumps(orgs))
        elif not orgs:
            print('No orgs.')
        else:
            for org in orgs:
                print_org(org)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

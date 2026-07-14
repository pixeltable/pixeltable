from __future__ import annotations

import json
import sys

from ..cloud import parse_org_uri, print_org
from ..http import get
from ..parser import Parser


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org status', description='Show status of an organization.')
    parser.add_argument('org_uri', help='Org URI: pxt://org')
    parser.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    args = parser.parse_args(argv)

    try:
        org_slug = parse_org_uri(args.org_uri, prog='pxt org status')
        resp = get(f'/api/cloud/orgs/{org_slug}')
        org = resp.get('org', resp) if isinstance(resp, dict) else {}
        if args.json_output:
            print(json.dumps(org))
        else:
            print_org(org)
    except SystemExit:
        raise
    except Exception as e:
        if args.json_output:
            print(json.dumps({'error': str(e)}), file=sys.stderr)
        else:
            print(f'pxt: error: {e}', file=sys.stderr)
        sys.exit(1)

"""`pxt org {list,status} [<uri>]` - manage organizations."""

from __future__ import annotations

import argparse
import json

from ..hosted import parse_org_uri, print_org
from ..parser import Parser
from ..utils import get_request

EPILOG = """\
Examples:
  pxt org list
  pxt org status pxt://org
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org', description='manage organizations', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('list', help='list organizations accessible to the current API key')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of an organization')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)


def _list(args: argparse.Namespace) -> None:
    resp = get_request('/api/orgs')
    orgs = resp.get('orgs', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(orgs))
    elif not orgs:
        print('No orgs.')
    else:
        for org in orgs:
            print_org(org)


def _status(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt org status')
    resp = get_request('/api/org', {'org': org})
    result = resp.get('org', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(result))
    else:
        print_org(result)

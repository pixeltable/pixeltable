"""`pxt org {create,use,list,status} [<uri>]` - manage organizations."""

from __future__ import annotations

import argparse
import json
import sys

from pixeltable.service import auth
from pixeltable.service.management_client import api_url

from ..hosted import parse_org_uri, print_org
from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt org create acme
  pxt org create acme --name "Acme Inc"
  pxt org use acme
  pxt org list
  pxt org status
  pxt org status pxt://org
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt org', description='manage organizations', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('create', help='create an organization, with its first database')
    p.add_argument('org', metavar='NAME', help='Namespace for the org; what pxt://org:db names')
    p.add_argument('--name', dest='display_name', help='Display name; defaults to NAME')
    p.add_argument('--location', help="e.g. 'aws/us-east-1'")
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('use', help='work in this organization from now on')
    p.add_argument('org', metavar='NAME', help='Organization to work in')

    p = sub.add_parser('list', help='list organizations accessible to the current API key')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of an organization')
    p.add_argument('org_uri', nargs='?', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'create':
        _create(args)
    elif args.action == 'use':
        _use(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)


def _create(args: argparse.Namespace) -> None:
    """Create the organization. Which one you are working in is `pxt org use`."""
    body = {'org_slug': args.org, 'display_name': args.display_name, 'location': args.location}
    resp = post_request('/api/orgs', {k: v for k, v in body.items() if v is not None})
    record = resp if isinstance(resp, dict) else {}

    if args.json_output:
        print(json.dumps(record))
        return
    print(f'{record.get("org_slug", args.org)}  (database {record.get("default_db_slug", "main")})')
    print(f'Run `pxt org use {record.get("org_slug", args.org)}` to work in it.')


def _use(args: argparse.Namespace) -> None:
    """Point this machine's session at one organization.

    The control plane authorizes from the token's own org claim, so switching means getting a new
    token rather than recording a preference.
    """
    resp = get_request('/api/orgs')
    orgs = resp.get('orgs', []) if isinstance(resp, dict) else []
    match = next((o for o in orgs if o.get('org') == args.org), None)
    if match is None:
        names = ', '.join(sorted(str(o.get('org')) for o in orgs)) or 'none'
        print(f'pxt org use: error: no organization named {args.org!r}. Yours: {names}', file=sys.stderr)
        sys.exit(1)

    auth.authorize_org(api_url(), str(match['org_id']))
    print(f'Working in {args.org}.')


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
    params = {} if args.org_uri is None else {'org': parse_org_uri(args.org_uri, prog='pxt org status')}
    resp = get_request('/api/org', params)
    result = resp.get('org', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(result))
    else:
        print_org(result)

"""`pxt org {create,list,status} [<uri>]` - manage organizations."""

from __future__ import annotations

import argparse
import json

from pixeltable.service import auth
from pixeltable.service.management_client import api_url

from ..hosted import parse_org_uri, print_org
from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt org create acme
  pxt org create acme --name "Acme Inc"
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

    p = sub.add_parser('list', help='list organizations accessible to the current API key')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of an organization')
    p.add_argument('org_uri', nargs='?', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'create':
        _create(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)


def _create(args: argparse.Namespace) -> None:
    """Create the organization, then put it on this machine's session.

    A token minted before the organization existed carries no claim for it, so nothing else would
    work until the next sign-in. Re-minting here is what makes `pxt db list` work on the next line.
    """
    body = {'org_slug': args.org, 'display_name': args.display_name, 'location': args.location}
    resp = post_request('/api/orgs', {k: v for k, v in body.items() if v is not None})
    record = resp if isinstance(resp, dict) else {}

    org_id = str(record.get('org_id') or '')
    if org_id:
        try:
            auth.authorize_org(api_url(), org_id)
        except auth.AuthError:
            # The organization exists either way; only this machine's session is behind, and the
            # next `pxt login` fixes it. Saying it failed would suggest otherwise.
            print('Created, but this session could not be updated. Run `pxt login` again.')

    if args.json_output:
        print(json.dumps(record))
        return
    print(f'{record.get("org_slug", args.org)}  (database {record.get("default_db_slug", "main")})')


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

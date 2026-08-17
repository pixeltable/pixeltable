"""`pxt secret {list,set,rm} <uri>` - manage org- and database-scoped runtime secrets."""

from __future__ import annotations

import argparse
import json
import sys

from pixeltable_cli.utils import split_pxt_uri

from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt secret list pxt://myorg
  pxt secret list pxt://myorg:mydb
  pxt secret set  pxt://myorg OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-...
  pxt secret rm   pxt://myorg:mydb OLD_KEY STALE_KEY

Secrets set on an org reach every database in it; secrets set on a database apply to that one and
win on a key collision. Each command restarts the pods that read them.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt secret', description='manage runtime secrets', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('list', help='list secret names in a scope (never their values)')
    p.add_argument('uri', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('set', help='add or replace secrets')
    p.add_argument('uri', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('assignments', nargs='+', metavar='KEY=VALUE')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('rm', help='remove secrets')
    p.add_argument('uri', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('keys', nargs='+', metavar='KEY')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'list':
        _list(args)
    elif args.action == 'set':
        _patch(args, _assignments(args.assignments), [])
    elif args.action == 'rm':
        _patch(args, {}, args.keys)


def _scope(uri: str, prog: str) -> tuple[str, str | None]:
    """Parse pxt://org or pxt://org:db; a db is the narrower scope, its absence means the whole org."""
    parts = split_pxt_uri(uri)
    if parts is None or parts.path is not None:
        print(f'{prog}: error: URI must be pxt://org or pxt://org:db, got {uri!r}', file=sys.stderr)
        sys.exit(2)
    return parts.org, parts.db


def _assignments(items: list[str]) -> dict[str, str]:
    secrets: dict[str, str] = {}
    for item in items:
        key, sep, value = item.partition('=')
        if not sep or key == '':
            print(f"pxt secret set: error: expected KEY=VALUE, got {item!r}", file=sys.stderr)
            sys.exit(2)
        secrets[key] = value
    return secrets


def _print_keys(keys: list[str], json_output: bool) -> None:
    if json_output:
        print(json.dumps(keys))
    elif not keys:
        print('No secrets.')
    else:
        for key in keys:
            print(key)


def _list(args: argparse.Namespace) -> None:
    org, db = _scope(args.uri, 'pxt secret list')
    params = {'org': org}
    if db is not None:
        params['db'] = db
    resp = get_request('/api/secrets', params)
    _print_keys(resp.get('keys', []) if isinstance(resp, dict) else [], args.json_output)


def _patch(args: argparse.Namespace, to_set: dict[str, str], to_delete: list[str]) -> None:
    org, db = _scope(args.uri, f'pxt secret {args.action}')
    resp = post_request('/api/secrets', {'org': org, 'db': db, 'set': to_set, 'delete': to_delete})
    _print_keys(resp.get('keys', []) if isinstance(resp, dict) else [], args.json_output)

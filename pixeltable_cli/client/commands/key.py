"""`pxt key {list,create,update,delete}` - manage the keys that reach your organization.

Two shapes share the command because they are the same thing to a user asking for a key:

  * a user key acts as you, reaching whatever you can reach. It is yours, it has no grants, and
    there is nothing to edit.
  * a runtime key acts as nobody. It belongs to the organization, reaches only what it is granted,
    and is what an agent or a job should hold instead of a copy of yours.

Passing --grant is what asks for the second.
"""

from __future__ import annotations

import argparse
import json
import sys

from pixeltable_cli.utils import split_pxt_uri

from ..parser import Parser
from ..utils import get_request, post_request

EPILOG = """\
Examples:
  pxt key create myci                                                # a key that acts as you
  pxt key create myapp  --grant access:pxt://myorg:db1/services/ingest
  pxt key create reader --grant access:pxt://myorg:db1/services      # call any service in db1
  pxt key create etl    --grant access:pxt://myorg:db1               # its services and its storage
  pxt key create ops    --grant manage:pxt://myorg:db1/services      # create/start/stop/delete them
  pxt key update myapp  --grant access:pxt://myorg:db2/services/reports \\
                        --revoke access:pxt://myorg:db1/services/ingest
  pxt key list
  pxt key delete myci

A grant is a verb on a pxt:// resource:

  access:pxt://org:db                   reach the data: every service in it, and its storage
  access:pxt://org:db/services          call any service in the database
  access:pxt://org:db/services/name     call that service - every route under it
  manage:pxt://org:db/services          create, list and manage any service in the database
  manage:pxt://org:db/services/name     start, stop, update, delete that service

access and manage are independent: neither implies the other. A key that can call a service cannot
reconfigure it, and one that can stop it cannot read what flows through it.

The organization must be your own - the one your key already belongs to. Keys are named uniquely
across both kinds, so a name always identifies one key.

The secret is printed once, by `create`, and cannot be retrieved afterwards. `update` edits grants
in place and leaves the secret alone, so widening or narrowing a key does not mean reissuing it.
"""

_VERBS = ('access', 'manage')
_GRANT = 'access|manage:pxt://org:db[/services[/name]]'


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt key', description='manage the keys that reach your organization', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('list', help="list your keys and the organization's runtime keys")
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('create', help='create a key; --grant makes it a scoped runtime key')
    p.add_argument('name', metavar='NAME', help='Name for the key, unique across both kinds')
    p.add_argument(
        '--grant',
        action='append',
        metavar=_GRANT,
        help='What the key may do. Omit for a key that acts as you (repeatable)',
    )
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('update', help='grant or revoke on a runtime key')
    p.add_argument('name', metavar='NAME', help='Name of the key to edit')
    p.add_argument('--grant', action='append', metavar=_GRANT, help='What to also allow (repeatable)')
    p.add_argument(
        '--revoke',
        action='append',
        metavar=_GRANT,
        help='What to stop allowing; must be something the key currently has (repeatable)',
    )
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete a key, revoking its secret')
    p.add_argument('name', metavar='NAME', help='Name of the key to delete')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)
    {'list': _list, 'create': _create, 'update': _update, 'delete': _delete}[args.action](args)


def _bad(flag: str, token: str) -> None:
    print(f'pxt key: error: {flag} takes {_GRANT}, got {token!r}', file=sys.stderr)
    sys.exit(2)


def _grants(values: list[str] | None, flag: str) -> list[str]:
    """Flatten repeated and comma-joined grants, matching --cols elsewhere in the CLI.

    Only the shape is checked here, so a typo is answered next to the flag that caused it. Which
    verbs a resource actually admits, and whether the organization is yours, are the control plane's
    to decide - it is the credential that settles which organization you are in.
    """
    out: list[str] = []
    for value in values or []:
        for part in value.split(','):
            spec = part.strip()
            verb, sep, uri = spec.partition(':')
            if not sep or verb not in _VERBS or not uri.startswith('pxt://'):
                _bad(flag, spec)
            parts = split_pxt_uri(uri)
            if parts is None or parts.db is None:
                _bad(flag, spec)
                return []  # unreachable; _bad exits
            path = (parts.path or '').strip('/')
            named_service = path.startswith('services/') and path.count('/') == 1 and path != 'services/'
            if path not in ('', 'services') and not named_service:
                _bad(flag, spec)
            if spec not in out:
                out.append(spec)
    return out


def _group(grants: list[str]) -> dict[str, list[str]]:
    """Group grants by database for display; the organization repeats on every one, the database does not."""
    out: dict[str, list[str]] = {}
    for spec in grants:
        verb, _sep, uri = spec.partition(':')
        parts = split_pxt_uri(uri)
        if parts is None or parts.db is None:
            continue
        path = (parts.path or '').strip('/')
        if path == '':
            what = 'all services and storage'
        elif path == 'services':
            what = 'all services'
        else:
            what = path[len('services/') :]
        out.setdefault(parts.db, []).append(f'{verb} {what}')
    return {db: sorted(items) for db, items in sorted(out.items())}


def _render(name: str, kind: str, grants: list[str] | None) -> None:
    """Grouped by database rather than one URI per line: the organization repeats on every grant, and
    what differs between them is the database, the verb, and what it applies to."""
    if kind == 'user':
        print(f'{name}  (acts as you: control plane and every database in the org)')
        return
    print(name)
    grouped = _group(grants or [])
    if not grouped:
        print('  (nothing)')
        return
    width = max(len(db) for db in grouped)
    for db in sorted(grouped):
        print(f'  {db + ":":<{width + 1}} {", ".join(grouped[db])}')


def _secret(value: str | None) -> None:
    if value:
        print(f'\nAPI key (shown once, store it now):\n{value}')


def _fetch() -> list[dict]:
    resp = get_request('/api/keys', {})
    return resp.get('keys', []) if isinstance(resp, dict) else []


def _show(key: dict, fallback_name: str = '') -> None:
    _render(key.get('name', fallback_name), key.get('key_type', 'runtime'), key.get('grants') or [])


def _list(args: argparse.Namespace) -> None:
    keys = _fetch()
    if args.json_output:
        print(json.dumps({'keys': keys}, indent=2))
        return
    if not keys:
        print('No keys.')
        return
    for key in keys:
        _show(key)


def _create(args: argparse.Namespace) -> None:
    # No --grant asks for a key that acts as you; any grant asks for one that acts as nobody.
    grants = _grants(args.grant, '--grant') if args.grant is not None else []
    resp = post_request('/api/keys', {'name': args.name, 'grants': grants})
    key = resp.get('key', {}) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(key, indent=2))
        return
    _show(key, args.name)
    _secret(key.get('api_key'))


def _update(args: argparse.Namespace) -> None:
    grant, revoke = _grants(args.grant, '--grant'), _grants(args.revoke, '--revoke')
    if not grant and not revoke:
        print('pxt key update: error: nothing to do; pass --grant and/or --revoke', file=sys.stderr)
        sys.exit(2)
    resp = post_request('/api/keys/update', {'name': args.name, 'allow': grant, 'revoke': revoke})
    key = resp.get('key', {}) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(key, indent=2))
        return
    _show(key, args.name)


def _delete(args: argparse.Namespace) -> None:
    post_request('/api/keys/delete', {'name': args.name})
    print(json.dumps({'name': args.name}) if args.json_output else args.name)

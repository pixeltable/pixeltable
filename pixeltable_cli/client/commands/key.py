"""`pxt key {list,create,update,delete}` - manage the keys that reach your organization.

Two shapes share the command because they are the same thing to a user asking for a key:

  * a user key acts as you, reaching whatever you can reach. It is yours, it has no grants, and
    there is nothing to edit.
  * a runtime key acts as nobody. It belongs to the organization, reaches only what it is granted,
    and is the one an agent or a job should use instead of a copy of yours.

Passing --grant asks for the second.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import NoReturn

from pixeltable_cli.utils import hosted_name_error, split_pxt_uri

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

A grant is a verb on a pxt:// resource: `access` to use it, `manage` to change it. The resource is
a database, its services, or one service, addressed by its base path and name. `manage` applies to
services only. `pxt key --help` lists the forms; the full table is in the CLI reference under
`pxt key`.

access and manage are independent: neither implies the other. A key that can call a service cannot
reconfigure it, and one that can stop it cannot read what flows through it.

The organization must be your own - the one your credential belongs to. Keys are named uniquely
across both kinds, so a name always identifies one key. Every member of the organization sees every
key, and can delete any of them.

Keys with grants are a preview: Pixeltable Cloud refuses them until they are enabled for your
organization.

The secret is printed once, by `create`, and cannot be retrieved afterwards. `update` edits grants
in place and leaves the secret alone, so widening or narrowing a key does not mean reissuing it.
"""

_VERBS = ('access', 'manage')
_GRANT = 'access|manage:pxt://org:db[/services[/path]]'


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt key', description='manage the keys that reach your organization', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('list', help="list every key in the organization, and each one's creator")
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


def _bad(flag: str, token: str, reason: str = '') -> NoReturn:
    detail = f': {reason}' if reason != '' else ''
    print(f'pxt key: error: {flag} takes {_GRANT}, got {token!r}{detail}', file=sys.stderr)
    sys.exit(2)


def _grant_scope(uri: str) -> tuple[str, str, str | None] | None:
    """The organization and database of a grant's URI, and the service path in the database; None for other URIs.

    The service path is None for the database itself, '' for all of its services, and base_path/name for
    one service, so it may have more than one component.
    """
    parts = split_pxt_uri(uri)
    if parts is None or parts.db is None:
        return None
    if parts.path is None:
        return parts.org, parts.db, None
    if parts.path == 'services':
        return parts.org, parts.db, ''
    head, sep, service = parts.path.partition('/')
    if head != 'services' or sep == '':
        return None
    if any(s in ('', '.', '..') or ':' in s for s in service.split('/')):
        return None
    return parts.org, parts.db, service


def _grants(values: list[str] | None, flag: str) -> list[str]:
    """Flatten repeated and comma-joined grants, matching --cols elsewhere in the CLI.

    Each one is checked against the forms the control plane accepts, so a typo is answered next to the
    flag that caused it. Whether the organization is yours is the control plane's to decide: the
    credential settles which organization you are in.
    """
    out: list[str] = []
    for value in values or []:
        for part in value.split(','):
            spec = part.strip()
            verb, sep, uri = spec.partition(':')
            scope = _grant_scope(uri) if sep != '' and verb in _VERBS else None
            if scope is None:
                _bad(flag, spec)
            org, db, service = scope
            name_error = hosted_name_error(org, 'organization name') or hosted_name_error(db, 'database name')
            if name_error is not None:
                _bad(flag, spec, name_error)
            if verb == 'manage' and service is None:
                _bad(flag, spec, 'manage applies to services, not to a database')
            if spec not in out:
                out.append(spec)
    return out


def _group(grants: list[str]) -> dict[str, list[str]]:
    """Group grants by database for display; the organization repeats on every one, the database does not."""
    out: dict[str, list[str]] = {}
    for spec in grants:
        verb, _sep, uri = spec.partition(':')
        scope = _grant_scope(uri)
        if scope is None:
            # unparseable here means the server granted something this version cannot read; saying
            # so beats leaving it out of what the key can do
            out.setdefault('?', []).append(spec)
            continue
        _org, db, service = scope
        what = 'all services and storage' if service is None else service or 'all services'
        out.setdefault(db, []).append(f'{verb} {what}')
    return {db: sorted(items) for db, items in sorted(out.items())}


def _render(name: str, kind: str, grants: list[str], created_by: str) -> None:
    """Print one key: its name, its creator when known, and its grants grouped by database.

    Grouped rather than one URI per line, since the organization repeats on every grant and only
    the database, the verb and the resource differ.
    """
    if kind == 'user':
        print(f'{name}  (acts as {created_by or "its creator"}: control plane and every database in the org)')
        return
    print(name if created_by == '' else f'{name}  (created by {created_by})')
    grouped = _group(grants)
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
    _render(
        key.get('name', fallback_name),
        key.get('key_type', 'runtime'),
        key.get('grants') or [],
        key.get('created_by') or '',
    )


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
    grants = _grants(args.grant, '--grant') if args.grant is not None else []
    resp = post_request('/api/key/create', {'name': args.name, 'grants': grants})
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
    resp = post_request('/api/key/update', {'name': args.name, 'allow': grant, 'revoke': revoke})
    key = resp.get('key', {}) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(key, indent=2))
        return
    _show(key, args.name)


def _delete(args: argparse.Namespace) -> None:
    post_request('/api/key/delete', {'name': args.name})
    print(json.dumps({'name': args.name}) if args.json_output else args.name)

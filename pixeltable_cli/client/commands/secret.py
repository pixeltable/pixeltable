"""`pxt secret {list,set,delete} <uri>` - manage org- and database-scoped runtime secrets."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from pixeltable_cli.utils import split_pxt_uri

from ..parser import Parser
from ..utils import EXIT_ERROR, RequestFailedError, get_request, post_request, print_aligned, print_next

EPILOG = """\
Examples:
  pxt secret list
  pxt secret list pxt://myorg
  pxt secret list pxt://myorg:mydb
  pxt secret set  pxt://myorg OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-...
  pxt secret delete pxt://myorg:mydb OLD_KEY STALE_KEY

An org secret applies to every database in the org; a database secret applies to that database and
wins on a key collision.

`list` without a URI lists the secrets of your credential's org and of every database in it; pxt://myorg
does the same for that org, and pxt://myorg:mydb lists the org's secrets and that database's.

After a `pxt secret set` or `pxt secret delete`, run `pxt db restart` to pick up the changes for a hosted database's
tables and `pxt service restart` to do the same for its services.
"""

SET_EPILOG = """\
Examples:
  pxt secret set pxt://myorg OPENAI_API_KEY=sk-...
  pxt secret set pxt://myorg:mydb OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-...

A process reads its secrets once, at startup, so a running one keeps the values it began with. Run
`pxt db restart` for a hosted database's tables and `pxt service restart` for its services.
"""

_OVERRIDES_ORG_NOTE = 'overrides an organization secret'

# the states in which a database or service has pods that have read their secrets; a service's failed roll keeps its
# previous pods serving when it can
_RUNNING = ('AVAILABLE', 'UPDATING')
_SERVICE_RUNNING = (*_RUNNING, 'FAILED')


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt secret', description="manage a database's secrets", epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('list', help='list the secret names of an org and its databases (never their values)')
    p.add_argument('uri', nargs='?', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('set', help='add or replace secrets (restart to pick them up)', epilog=SET_EPILOG)
    p.add_argument('uri', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('assignments', nargs='+', metavar='KEY=VALUE')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete secrets')
    p.add_argument('uri', help='Scope URI: pxt://org or pxt://org:db')
    p.add_argument('keys', nargs='+', metavar='KEY')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'list':
        _list(args)
    elif args.action == 'set':
        _set(args)
    elif args.action == 'delete':
        _delete(args)


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
            print(f'pxt secret set: error: expected KEY=VALUE, got {item!r}', file=sys.stderr)
            sys.exit(2)
        secrets[key] = value
    return secrets


def _print_secrets(org: str, secrets: list[dict[str, Any]], json_output: bool, with_overrides: bool) -> None:
    """secrets are dicts with 'key' and 'db', where db is None for an org secret.

    with_overrides marks db secrets that override an org secret; this requires secrets to include all org secrets.
    """
    org_uri = f'pxt://{org}'
    secrets = sorted(secrets, key=lambda s: (s['db'] is not None, s['db'] or '', s['key']))
    org_keys = {s['key'] for s in secrets if s['db'] is None}
    rows: list[dict[str, Any]] = []
    for s in secrets:
        if s['db'] is None:
            rows.append({'key': s['key'], 'scope': org_uri})
        else:
            row = {'key': s['key'], 'scope': f'{org_uri}:{s["db"]}'}
            if with_overrides and s['key'] in org_keys:
                row['overrides_org'] = True
            rows.append(row)
    if json_output:
        print(json.dumps(rows))
    elif with_overrides:
        table = [[r['key'], r['scope'], _OVERRIDES_ORG_NOTE if 'overrides_org' in r else ''] for r in rows]
        print_aligned(['KEY', 'SCOPE', 'NOTE'], table, right_align=set())
    else:
        print_aligned(['KEY', 'SCOPE'], [[r['key'], r['scope']] for r in rows], right_align=set())


def _list(args: argparse.Namespace) -> None:
    params = {}
    db = None
    if args.uri is not None:
        org, db = _scope(args.uri, 'pxt secret list')
        params['org'] = org
        if db is not None:
            params['db'] = db
    resp = get_request('/api/secrets', params)
    if len(resp['secrets']) == 0 and not args.json_output:
        scope = f'pxt://{resp["org"]}' if db is None else f'pxt://{resp["org"]}:{db}'
        print(f'No secrets for {scope}.')
        return
    _print_secrets(resp['org'], resp['secrets'], args.json_output, with_overrides=True)


def _set(args: argparse.Namespace) -> None:
    org, db = _scope(args.uri, 'pxt secret set')
    secrets = _assignments(args.assignments)
    _change(org, db, [('/api/secrets', {'org': org, 'db': db, 'key': k, 'value': v}) for k, v in secrets.items()], args)


def _delete(args: argparse.Namespace) -> None:
    org, db = _scope(args.uri, 'pxt secret delete')
    _change(org, db, [('/api/secrets/delete', {'org': org, 'db': db, 'key': k}) for k in args.keys], args)


def _change(org: str, db: str | None, requests: list[tuple[str, dict[str, Any]]], args: argparse.Namespace) -> None:
    """Sends each change in turn and reports the keys changed, with the restarts they need, even when one fails."""
    changed: list[str] = []
    failure: RequestFailedError | None = None
    for path, body in requests:
        try:
            post_request(path, body, raise_on_error=True)
        except RequestFailedError as e:
            failure = e
            break
        changed.append(body['key'])
    if len(changed) > 0:
        _print_secrets(org, [{'key': key, 'db': db} for key in changed], args.json_output, with_overrides=False)
        _print_restarts(org, db)
    if failure is not None:
        failure.report()
        sys.exit(EXIT_ERROR)


def _print_restarts(org: str, db: str | None) -> None:
    """Names a restart for each running database and service that reads the changed secrets.

    A pod reads its secrets when it starts, so a running one keeps the old values until restarted; a stopped one
    reads the new values when it next starts, so it needs nothing.
    """
    sys.stdout.flush()
    try:
        if db is None:
            reports = get_request('/api/dbs', {'org': org}, raise_on_error=True).get('reports', [])
        else:
            # a database that does not exist yet has no pods to restart
            resp = get_request('/api/db', {'org': org, 'db': db, 'missing_ok': True}, raise_on_error=True)
            reports = [resp.get('report') or {}]
        dbs = [r['db'] for r in reports if 'db' in r]
        running = {r['db'] for r in reports if (r.get('current') or {}).get('state') in _RUNNING}
        # a service runs in pods of its own, which stopping its database leaves running
        services = {
            d: get_request('/api/service/list', {'target': f'pxt://{org}:{d}'}, raise_on_error=True) for d in dbs
        }
    except RequestFailedError as e:
        print(
            f'pxt: could not list what reads these secrets ({e}); restart each running database and service',
            file=sys.stderr,
        )
        return
    for d in dbs:
        if d in running:
            print_next(f'pxt db restart pxt://{org}:{d}', 'restarts its tables, so they read the new values')
        for service in services[d]:
            if service['state'] in _SERVICE_RUNNING:
                where = f'{service["catalog_path"]}/{service["name"]}'
                print_next(f'pxt service restart {where}', 'restarts it, so it reads the new values')

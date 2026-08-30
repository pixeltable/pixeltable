"""`pxt db {diff,update,list,status,start,stop,build-image,delete}` - manage hosted databases."""

from __future__ import annotations

import argparse
import json
import sys

from ...types import DbChangeOp, DbPlan, Resolution
from ..hosted import exit_unless_reached, parse_org_uri, poll_db, print_db, resolve_db_uri, spinner
from ..parser import Parser
from ..utils import EXIT_CHANGES_PENDING, EXIT_IN_AGREEMENT, EXIT_REFUSED, confirm_or_exit, get_request, post_request

EPILOG = """\
Examples:
  pxt db diff pxt://org:db     # what update would change; exit 2 if anything is pending
  pxt db update pxt://org:db   # apply it: secrets, then the artifacts, then capacity
  pxt db list pxt://org
  pxt db status pxt://org:db
  pxt db start pxt://org:db
  pxt db stop pxt://org:db
  pxt db build-image pxt://org:db   # build an image without comparing first
  pxt db delete pxt://org:db

The uri selects the matching [[pixeltable.database]] entry in the project configuration:

  [[pixeltable.database]]
  name = 'pxt://org:db'      # what 'pxt db update pxt://org:db' looks for

The entry says which of the project's files the database gets (include/exclude), what the image
holds (system_dependencies, python_version), what the database runs on (cpu, memory_mb, disk_gb, workers)
and which secrets it holds. 'diff' compares the entry against the database; 'update' applies the difference.

Exit status of diff and update: 0 in agreement, 2 changes pending, 3 refused, 1 error.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db', description='manage hosted databases', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    for verb in ('diff', 'update'):
        p = sub.add_parser(verb, help=f'{"show" if verb == "diff" else "apply"} what the project declares')
        p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
        p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
        if verb == 'update':
            p.add_argument('-f', '--force', action='store_true', help='skip confirmation')
            p.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
            p.add_argument(
                '--allow-destructive',
                action='store_true',
                dest='allow_destructive',
                help='permit changes that take capacity away or delete a secret',
            )

    p = sub.add_parser('list', help='list hosted databases for an org')
    p.add_argument('org_uri', help='Org URI: pxt://org')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('start', help='start (wake) a stopped hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('stop', help='stop (sleep) a running hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('build-image', help='build the image a hosted database runs on, from a project')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('delete', help='delete a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    args = parser.parse_args(argv)

    if args.action == 'diff':
        _diff(args)
    elif args.action == 'update':
        _update(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)
    elif args.action == 'start':
        _start(args)
    elif args.action == 'stop':
        _stop(args)
    elif args.action == 'build-image':
        _build_image(args)
    elif args.action == 'delete':
        _delete(args)


def _list(args: argparse.Namespace) -> None:
    org = parse_org_uri(args.org_uri, prog='pxt db list')
    resp = get_request('/api/dbs', {'org': org})
    dbs = resp.get('databases', []) if isinstance(resp, dict) else []
    if args.json_output:
        print(json.dumps(dbs))
    elif not dbs:
        print('No databases.')
    else:
        for db in dbs:
            print_db(db)


def _status(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db status')
    resp = get_request('/api/db', {'org': org, 'db': db})
    result = resp.get('database', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)


def _start(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db start')
    post_request('/api/db/start', {'org': org, 'db': db})
    result = poll_db(org, db, {'UPDATING', 'STARTING'}, f"Database '{db}' is starting...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'AVAILABLE', f'starting database {db!r}')


def _stop(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db stop')
    post_request('/api/db/stop', {'org': org, 'db': db})
    result = poll_db(org, db, {'STOPPING'}, f"Database '{db}' is stopping...")
    if args.json_output:
        print(json.dumps(result))
    else:
        print_db(result)
    exit_unless_reached(result, 'STOPPED', f'stopping database {db!r}')


def _db_uri(args: argparse.Namespace, prog: str) -> str:
    """The uri the verb acts on, defaulting to the one the config names."""
    org, db = resolve_db_uri(args.db_uri, prog=prog)
    return f'pxt://{org}:{db}'


def _diff(args: argparse.Namespace) -> None:
    plan = DbPlan.model_validate(post_request('/api/db/diff', {'db_uri': _db_uri(args, 'pxt db diff')}))
    _print_plan(plan, as_json=args.json_output)
    sys.exit(EXIT_IN_AGREEMENT if plan.in_agreement else EXIT_CHANGES_PENDING)


def _update(args: argparse.Namespace) -> None:
    body = {'db_uri': _db_uri(args, 'pxt db update')}
    plan = DbPlan.model_validate(post_request('/api/db/diff', body))
    if plan.in_agreement:
        _print_plan(plan, as_json=args.json_output)
        sys.exit(EXIT_IN_AGREEMENT)
    if args.dry_run:
        _print_plan(plan, as_json=args.json_output)
        sys.exit(EXIT_CHANGES_PENDING)

    if not args.json_output:
        # the pending plan, for the confirmation that follows; --json emits the applied plan alone
        _print_plan(plan, as_json=False)
    what = f'{plan.summary.ops} change(s)' if plan.exists else f'create {plan.db_uri} and apply it'
    minutes = ', which rebuilds the image and takes several minutes' if plan.summary.rebuild else ''
    confirm_or_exit(
        f'apply {what} to {plan.db_uri}{minutes}?',
        args.force,
        refused_exit_code=EXIT_REFUSED,
        on_refusal=lambda: _print_plan(plan, as_json=args.json_output),
    )

    label = None if args.json_output else f'Updating {plan.db_uri} ...'
    with spinner(label):
        applied = DbPlan.model_validate(
            post_request('/api/db/update', {**body, 'allow_destructive': args.allow_destructive})
        )
    _print_plan(applied, as_json=args.json_output, applied=True)
    if not applied.in_agreement:
        # an operation nothing applies, such as a placement change, leaves the database out of agreement
        sys.exit(EXIT_CHANGES_PENDING)


_MARKERS: dict[Resolution, str] = {
    'up_to_date': '=',
    'create': '+',
    'update_additive': '~',
    'update_destructive': '~',
    'unsupported': '!',
}
_PENDING: dict[Resolution, str] = {
    'up_to_date': 'up to date',
    'create': 'will be created',
    'update_additive': 'will be updated',
    'update_destructive': 'will be updated (destructive)',
    'unsupported': 'declares what cannot be changed',
}


def _print_plan(plan: DbPlan, *, as_json: bool, applied: bool = False) -> None:
    if as_json:
        print(plan.model_dump_json(indent=2))
        return
    resolution = plan.resolution
    state = (plan.status or _PENDING[resolution]) if applied else _PENDING[resolution]
    print(f'{_MARKERS[resolution]} {plan.db_uri:<28s} {state}  {plan.state or "absent"}')
    for op in plan.ops:
        print(f'    {op.description}  [{op.severity}]')
    s = plan.summary
    print()
    print(f'Plan: {s.ops} change(s), {s.destructive} destructive, {s.unsupported} unsupported')


def _delete(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db delete')
    post_request('/api/db/delete', {'org': org, 'db': db})
    if args.json_output:
        print(json.dumps({'deleted': db}))
    else:
        print(f"Deleted database '{db}'.")


def _build_image(args: argparse.Namespace) -> None:
    db_uri = _db_uri(args, 'pxt db build-image')
    label = (
        None
        if args.json_output
        else 'Uploading the project files and building the image (this may take 10 minutes or longer) ...'
    )
    with spinner(label):
        ops = [DbChangeOp.model_validate(op) for op in post_request('/api/db/build-image', {'db_uri': db_uri})]
    if args.json_output:
        print(json.dumps([op.model_dump(mode='json') for op in ops]))
    else:
        print(f'Uploaded the project files to {db_uri} and rebuilt its image.')

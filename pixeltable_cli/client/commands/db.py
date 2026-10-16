"""`pxt db {diff,update,list,status,logs,start,stop,restart,retry,build-image,delete}` - manage hosted databases."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import pydantic

from ...models import DbBuildImageResponse, DbLifecycleResponse
from ...types import DbPlan, GenerationReceipt, Resolution
from ..hosted import (
    add_logs_args,
    await_receipts,
    describe_receipt,
    exit_unless_observed,
    parse_db_uri,
    print_db,
    print_logs,
    print_receipt,
    resolve_db_uri,
    spinner,
)
from ..parser import Parser
from ..utils import (
    EXIT_CHANGES_PENDING,
    EXIT_IN_AGREEMENT,
    EXIT_REFUSED,
    confirm_or_exit,
    get_request,
    post_request,
    print_json_schema,
)

EPILOG = """\
Examples:
  pxt db diff pxt://org:db     # what update would change; exit 2 if anything is pending
  pxt db diff --json-schema    # the schema of the --json output, on its own
  pxt db update pxt://org:db   # apply it: the artifacts, then capacity
  pxt db update pxt://org:db --no-wait   # submit it and return without waiting
  pxt db list
  pxt db status pxt://org:db
  pxt db status --json-schema  # the schema of its --json output, on its own
  pxt db logs pxt://org:db              # what the database's pod logged in the last hour
  pxt db logs pxt://org:db --since 10m --tail 50
  pxt db start pxt://org:db
  pxt db stop pxt://org:db
  pxt db restart pxt://org:db   # cycle its pods onto the image and project it runs
  pxt db retry pxt://org:db     # try a failed update again
  pxt db build-image pxt://org:db   # build an image without comparing first
  pxt db delete pxt://org:db -f   # no confirmation

The uri selects the matching [[pixeltable.database]] entry in the project configuration:

  [[pixeltable.database]]
  name = 'pxt://org:db'      # what 'pxt db update pxt://org:db' looks for

The entry says which of the project's files the database gets (include/exclude), what goes into
the image (system_dependencies, python_version), and what the database runs on (cpu, memory_mb,
disk_gb, workers). 'diff' compares the entry against the database; 'update' applies the difference.
Secrets are set separately, with 'pxt secret'.

Pixeltable Cloud applies an accepted update, start, stop, restart, retry, image build or delete even if the
command exits early: Ctrl-C only stops the waiting, and --no-wait returns once the change is accepted.
Rerunning the same update reports the change already accepted, and retries it if it failed.

Exit status of diff and update: 0 in agreement, 2 changes pending, 3 refused, 1 error.
"""


def run(argv: list[str]) -> None:
    parser = Parser(prog='pxt db', description='manage hosted databases', epilog=EPILOG)
    sub = parser.add_subparsers(dest='action', required=True)

    if argv == ['diff', '--json-schema']:
        print_json_schema(pydantic.TypeAdapter(DbPlan))
        return

    if argv == ['status', '--json-schema']:
        # local import: management_protocol pulls in pixeltable, which is time-consuming to import
        from pixeltable.service.management_protocol import DatabaseReport

        print_json_schema(pydantic.TypeAdapter(DatabaseReport))
        return

    for verb in ('diff', 'update'):
        p = sub.add_parser(verb, help=f'{"show" if verb == "diff" else "apply"} what the project defines')
        p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
        p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
        if verb == 'update':
            p.add_argument('-f', '--force', action='store_true', help='skip confirmation')
            p.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
            p.add_argument(
                '--allow-destructive',
                action='store_true',
                dest='allow_destructive',
                help='permit changes that take capacity away',
            )
            _add_no_wait(p)

    p = sub.add_parser('list', help='list hosted databases')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('status', help='show status of a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')

    p = sub.add_parser('logs', help="read the database pod's log")
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    add_logs_args(p)

    for verb, help_text in (
        ('start', 'start (wake) a stopped hosted database'),
        ('stop', 'stop (sleep) a running hosted database'),
        ('restart', 'restart a hosted database'),
        ('retry', 'retry the failed current update of a hosted database'),
    ):
        p = sub.add_parser(verb, help=help_text)
        p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
        p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
        _add_no_wait(p)

    p = sub.add_parser('build-image', help='build the image a hosted database runs on, from a project')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    _add_no_wait(p)

    p = sub.add_parser('delete', help='delete a hosted database')
    p.add_argument('db_uri', nargs='?', help='Database URI: pxt://org:db (default: db_uri from the config)')
    p.add_argument('-f', '--force', action='store_true', help='skip confirmation')
    p.add_argument('--json', action='store_true', dest='json_output', help='Emit JSON output')
    _add_no_wait(p)

    args = parser.parse_args(argv)

    if args.action == 'diff':
        _diff(args)
    elif args.action == 'update':
        _update(args)
    elif args.action == 'list':
        _list(args)
    elif args.action == 'status':
        _status(args)
    elif args.action == 'logs':
        _logs(args)
    elif args.action in ('start', 'stop', 'restart', 'retry'):
        _change_lifecycle(args)
    elif args.action == 'build-image':
        _build_image(args)
    elif args.action == 'delete':
        _delete(args)


def _list(args: argparse.Namespace) -> None:
    resp = get_request('/api/dbs')
    dbs = resp.get('reports', []) if isinstance(resp, dict) else []
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
    report = resp.get('report', resp) if isinstance(resp, dict) else {}
    if args.json_output:
        print(json.dumps(report))
    else:
        print_db(report, resp.get('worker_status'))


def _logs(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db logs')
    print_logs({'org': org, 'db': db}, args)


def _add_no_wait(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--no-wait',
        action='store_false',
        dest='wait',
        help='return once Pixeltable Cloud accepts the change, without waiting for it to finish',
    )


def _change_lifecycle(args: argparse.Namespace) -> None:
    db_uri = _db_uri(args, f'pxt db {args.action}')
    with spinner(None if args.json_output else f'Submitting the {args.action} of {db_uri} ...'):
        accepted = DbLifecycleResponse.model_validate(
            post_request(f'/api/db/{args.action}', {'db_uri': db_uri, 'wait': False})
        )
    settled = _await(db_uri, [accepted.receipt], args)
    report, workers = (accepted.report, accepted.worker_status) if not args.wait else _db_report(db_uri)
    if args.wait and settled[0].observed and len(report) == 0:
        if args.json_output:
            print(json.dumps({'deleted': db_uri, 'receipt': settled[0].model_dump(mode='json')}))
        else:
            print(f'Deleted {db_uri}.')
        return
    if args.json_output:
        print(json.dumps(report))
    else:
        print_db(report, workers)
        _print_receipts(db_uri, settled, waited=args.wait)
    exit_unless_observed(settled, retry_command=f'pxt db retry {db_uri}')


def _db_uri(args: argparse.Namespace, prog: str) -> str:
    """The uri the verb acts on, defaulting to the one the config names."""
    org, db = resolve_db_uri(args.db_uri, prog=prog)
    return f'pxt://{org}:{db}'


def _diff(args: argparse.Namespace) -> None:
    plan = DbPlan.model_validate(post_request('/api/db/diff', {'db_uri': _db_uri(args, 'pxt db diff')}))
    _print_plan(plan, as_json=args.json_output)
    sys.exit(EXIT_IN_AGREEMENT if plan.in_agreement else EXIT_CHANGES_PENDING)


def _update(args: argparse.Namespace) -> None:
    body: dict[str, Any] = {'db_uri': _db_uri(args, 'pxt db update')}
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
        # text mode printed the pending plan above; --json skipped it, so a refusal still emits it
        on_refusal=lambda: _print_plan(plan, as_json=True) if args.json_output else None,
    )

    update = {
        **body,
        'allow_destructive': args.allow_destructive,
        'expected_generation': plan.generation,
        'wait': False,
    }
    with spinner(None if args.json_output else f'Submitting the update of {plan.db_uri} ...'):
        applied = DbPlan.model_validate(post_request('/api/db/update', update))
    applied.receipts = _await(plan.db_uri, applied.receipts, args)
    if args.wait and all(r.observed for r in applied.receipts):
        for op in applied.ops:
            op.status = 'applied'
        applied.status = 'applied'
        applied.resolution = 'up_to_date'
        applied.state = (_db_report(plan.db_uri)[0].get('current') or {}).get('state')
    _print_plan(applied, as_json=args.json_output, applied=True)
    if not args.json_output:
        _print_receipts(plan.db_uri, applied.receipts, waited=args.wait)
    exit_unless_observed(applied.receipts, retry_command=f'pxt db retry {plan.db_uri}')


def _print_receipts(db_uri: str, receipts: list[GenerationReceipt], *, waited: bool) -> None:
    for receipt in receipts:
        print_receipt(receipt)
    if not waited and len(receipts) > 0:
        print(f'\nThe change continues in Pixeltable Cloud; run `pxt db status {db_uri}` to check its progress.')


def _await(db_uri: str, receipts: list[GenerationReceipt], args: argparse.Namespace) -> list[GenerationReceipt]:
    """The receipts once settled, or as accepted under --no-wait."""
    if not args.wait:
        return receipts
    return await_receipts(db_uri, receipts, show=not args.json_output, status_command=f'pxt db status {db_uri}')


def _db_report(db_uri: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The database's report and its pods; an empty report once the database is deleted."""
    org, db = parse_db_uri(db_uri)
    resp = get_request('/api/db', {'org': org, 'db': db, 'missing_ok': True})
    return resp.get('report') or {}, resp.get('worker_status') or []


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
    'unsupported': 'defines what cannot be changed',
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
    if not applied and not any(op.target == 'generation' for op in plan.ops):
        for receipt in plan.receipts:
            print(f'    the current generation {describe_receipt(receipt)}; this update replaces it')
    s = plan.summary
    print()
    print(f'Plan: {s.ops} change(s), {s.destructive} destructive')


def _delete(args: argparse.Namespace) -> None:
    org, db = resolve_db_uri(args.db_uri, prog='pxt db delete')
    db_uri = f'pxt://{org}:{db}'
    confirm_or_exit(f'delete {db_uri}? This is irreversible.', args.force, refused_exit_code=EXIT_REFUSED)
    with spinner(None if args.json_output else f'Submitting the deletion of {db_uri} ...'):
        accepted = DbLifecycleResponse.model_validate(post_request('/api/db/delete', {'db_uri': db_uri, 'wait': False}))
    settled = _await(db_uri, [accepted.receipt], args)
    exit_unless_observed(settled, retry_command=f'pxt db retry {db_uri}')
    deleted = settled[0].observed
    if args.json_output:
        print(json.dumps({'deleted' if deleted else 'deleting': db, 'receipt': settled[0].model_dump(mode='json')}))
    elif deleted:
        print(f"Deleted database '{db}'.")
    else:
        print(f"Deleting database '{db}'; its name stays taken until its teardown finishes.")
        _print_receipts(db_uri, settled, waited=args.wait)


def _build_image(args: argparse.Namespace) -> None:
    db_uri = _db_uri(args, 'pxt db build-image')
    with spinner(None if args.json_output else f'Submitting an image build for {db_uri} ...'):
        resp = DbBuildImageResponse.model_validate(
            post_request('/api/db/build-image', {'db_uri': db_uri, 'wait': False})
        )
    settled = _await(db_uri, resp.receipts, args)
    ops = resp.ops
    if args.wait and all(r.observed for r in settled):
        for op in ops:
            if op.target == 'image':
                op.status = 'applied'
    if args.json_output:
        print(json.dumps([op.model_dump(mode='json') for op in ops]))
    else:
        statuses = {op.target: op.status for op in ops}
        archive = (
            'uploaded the project files' if statuses.get('archive') == 'applied' else 'reused the existing archive'
        )
        image = {'applied': 'rebuilt its image', 'accepted': 'is rebuilding its image'}.get(
            statuses.get('image') or '', 'reused the existing image'
        )
        print(f'{db_uri}: {archive}, {image}.')
    exit_unless_observed(settled, retry_command=f'pxt db retry {db_uri}')

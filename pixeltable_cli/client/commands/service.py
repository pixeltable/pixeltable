"""`pxt service {diff,update,prune,stop,list,example}` - run the services an application file declares."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...service_types import ServiceChangeOp, ServiceDeployment, ServicePlan, ServiceResolution, delete_service_op
from ...utils import PxtPath
from ..confirm import confirm_or_exit
from ..parser import Parser
from ..utils import get_request, post_request

_EXAMPLE_APP = '''\
"""Pixeltable application, written by 'pxt service example'.

A service is a FastAPIRouter with routes declared against the models in a schema file. The models name
tables; the target given to 'pxt service update' says which directory those tables live in, so the same
file can be served against a development directory and a production one.

    pxt schema update schema.py TARGET     # create the tables the routes need
    pxt service update app.py TARGET       # serve this file's services against them
"""

from __future__ import annotations  # required to declare a model on Python 3.14+

import pixeltable as pxt
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    doc_id: pxt.Int
    title: pxt.String
    body: pxt.String | None


# the router names the service; without name= it takes the name of the variable holding it
ingest = FastAPIRouter(name='ingest')

# POST /docs inserts a row and returns the columns named in outputs
ingest.add_insert_route(Docs, path='/docs', inputs=['doc_id', 'title', 'body'], outputs=['doc_id'])

# GET /docs/search returns rows; a query route serves a @pxt.query function
ingest.add_compute_route(Docs, path='/titles', inputs=['title'], outputs=['title'])
'''

DIFF_EPILOG = """\
Examples:
  pxt service diff app.py my_dir          # what update would change; exit 2 if anything is pending
  pxt service diff app.py my_dir --json
"""

UPDATE_EPILOG = """\
Examples:
  pxt service update app.py my_dir                       # start what is declared, restart what changed
  pxt service update app.py my_dir --allow-destructive   # also stop serving routes that changed or went away
"""

PRUNE_EPILOG = """\
Examples:
  pxt service prune app.py my_dir     # stop and forget deployments the file does not declare

A stopped service can be started again with 'pxt service update'.
"""

STOP_EPILOG = """\
Examples:
  pxt service stop ingest             # a bare name, when only one deployment has it
  pxt service stop my_dir/ingest      # the deployment of that name under my_dir
  pxt service stop ingest reader
"""

LIST_EPILOG = """\
Examples:
  pxt service list                    # every service running locally
  pxt service list my_dir             # those bound at my_dir and below it
"""

VERBS = ('diff', 'update', 'prune', 'stop', 'list', 'example')

EXIT_IN_AGREEMENT = 0
EXIT_ERROR = 1
EXIT_CHANGES_PENDING = 2
EXIT_REFUSED = 3

_MARKERS: dict[ServiceResolution, str] = {
    'up_to_date': '=',
    'create': '+',
    'update_additive': '~',
    'update_destructive': '~',
    'unsupported': '!',
    'blocked': '!',
}
_PENDING: dict[ServiceResolution, str] = {
    'up_to_date': 'up to date',
    'create': 'will be started',
    'update_additive': 'will be restarted',
    'update_destructive': 'will be restarted (destructive)',
    'unsupported': 'cannot be served',
    'blocked': 'blocked: the database has to change first',
}


def run(argv: list[str]) -> None:
    if len(argv) == 0 or argv[0] in ('-h', '--help'):
        print(
            'usage: pxt service <verb> APP TARGET [options]\n\nverbs:\n'
            '  diff     show the changes that update would make; exit 2 if any are pending\n'
            '  update   start the services APP declares against TARGET, and restart the ones that changed\n'
            '  prune    stop and forget the deployments at TARGET that APP does not declare\n'
            '  stop     stop the named services\n'
            '  list     what is running locally, and where\n'
            '  example  write a working application file to start from\n\n'
            'APP is a Python file declaring FastAPIRouter services; TARGET is the catalog directory their\n'
            "models bind against. Run 'pxt service example' for a file to start from."
        )
        sys.exit(EXIT_IN_AGREEMENT if len(argv) > 0 else EXIT_ERROR)

    verb = argv[0]
    if verb not in VERBS:
        print(f'pxt service: unknown verb: {verb} (available: {", ".join(VERBS)})', file=sys.stderr)
        sys.exit(EXIT_ERROR)

    if verb == 'example':
        ap = Parser(prog='pxt service example', usage_exit_code=EXIT_ERROR)
        ap.add_argument('--out', help='write to this file instead of standard output')
        args = ap.parse_args(argv[1:])
        _example(args.out)
        return

    if verb == 'stop':
        ap = Parser(prog='pxt service stop', epilog=STOP_EPILOG, usage_exit_code=EXIT_ERROR)
        ap.add_argument('names', nargs='+', help='service names, or TARGET/NAME to disambiguate')
        ap.add_argument('--json', action='store_true', dest='as_json')
        args = ap.parse_args(argv[1:])
        _stop(args.names, as_json=args.as_json)
        return

    if verb == 'list':
        ap = Parser(prog='pxt service list', epilog=LIST_EPILOG, usage_exit_code=EXIT_ERROR)
        ap.add_argument('target', nargs='?', default=None, help='report only the services bound here and below')
        ap.add_argument('--json', action='store_true', dest='as_json')
        args = ap.parse_args(argv[1:])
        _list(args.target, as_json=args.as_json)
        return

    epilogs = {'diff': DIFF_EPILOG, 'update': UPDATE_EPILOG, 'prune': PRUNE_EPILOG}
    # a usage error exits EXIT_ERROR, not argparse's 2, which here means that changes are pending
    ap = Parser(prog=f'pxt service {verb}', epilog=epilogs[verb], usage_exit_code=EXIT_ERROR)
    ap.add_argument('app', help='path to a Python file declaring services')
    ap.add_argument('target', help='catalog directory the services bind against')
    ap.add_argument('--json', action='store_true', dest='as_json')
    if verb in ('update', 'prune'):
        ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
        ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    if verb == 'update':
        ap.add_argument(
            '--allow-destructive',
            action='store_true',
            dest='allow_destructive',
            help='permit changes that stop serving a route callers may be using',
        )
    args = ap.parse_args(argv[1:])

    path = Path(args.app)
    if not path.is_file():
        print(
            f'pxt service {verb}: application file not found: {args.app}\n'
            "run 'pxt service example' for a file to start from",
            file=sys.stderr,
        )
        sys.exit(EXIT_ERROR)
    app_file = str(path.resolve())

    if verb == 'diff':
        _diff(app_file, args.target, as_json=args.as_json)
    elif verb == 'prune':
        _prune(app_file, args.target, as_json=args.as_json, force=args.force, dry_run=args.dry_run)
    else:
        _update(
            app_file,
            args.target,
            as_json=args.as_json,
            force=args.force,
            dry_run=args.dry_run,
            allow_destructive=args.allow_destructive,
        )


def _example(out: str | None) -> None:
    if out is None:
        sys.stdout.write(_EXAMPLE_APP)
        return
    Path(out).write_text(_EXAMPLE_APP, encoding='utf-8')
    print(f'wrote {out}')


def _service_plan(app_file: str, target: PxtPath) -> ServicePlan:
    plan: ServicePlan = post_request('/api/localservice/diff', {'app_file': app_file, 'target': target})
    return plan


def _diff(app_file: str, target: PxtPath, *, as_json: bool) -> None:
    plan = _service_plan(app_file, target)
    _print_plan(plan, as_json=as_json)
    sys.exit(EXIT_IN_AGREEMENT if plan['in_agreement'] else EXIT_CHANGES_PENDING)


def _update(
    app_file: str, target: PxtPath, *, as_json: bool, force: bool, dry_run: bool, allow_destructive: bool
) -> None:
    plan = _service_plan(app_file, target)
    if plan['in_agreement']:
        _print_plan(plan, as_json=as_json)
        sys.exit(EXIT_IN_AGREEMENT)
    if dry_run:
        _print_plan(plan, as_json=as_json)
        sys.exit(EXIT_CHANGES_PENDING)

    s = plan['summary']
    restarts = f', interrupting {s["restarts"]} running service(s)' if s['restarts'] > 0 else ''
    confirm_or_exit(
        f'start or restart {s["create"] + s["update_additive"] + s["update_destructive"]} service(s){restarts}?',
        force,
        refused_exit_code=EXIT_REFUSED,
        on_refusal=lambda: _print_plan(plan, as_json=as_json),
    )

    applied: ServicePlan = post_request(
        '/api/localservice/update', {'app_file': app_file, 'target': target, 'allow_destructive': allow_destructive}
    )
    _print_plan(applied, as_json=as_json, applied=True)


def _prune(app_file: str, target: PxtPath, *, as_json: bool, force: bool, dry_run: bool) -> None:
    plan = _service_plan(app_file, target)
    extras = plan['extras']
    if len(extras) == 0:
        if as_json:
            print(json.dumps({**plan, 'ops': []}, indent=2))
        else:
            print('nothing to prune')
        sys.exit(EXIT_IN_AGREEMENT)

    if dry_run:
        _print_ops([delete_service_op(name, None, 'skipped') for name in extras], as_json=as_json, verb='would stop')
        sys.exit(EXIT_CHANGES_PENDING)

    confirm_or_exit(
        f'stop {len(extras)} service(s) the file does not declare?',
        force,
        refused_exit_code=EXIT_REFUSED,
        on_refusal=lambda: _print_ops(
            [delete_service_op(name, None, 'refused') for name in extras], as_json=as_json, verb='would stop'
        ),
    )
    pruned: ServicePlan = post_request('/api/localservice/prune', {'app_file': app_file, 'target': target})
    _print_ops(pruned.get('ops', []), as_json=as_json, verb='stopped')


def _stop(names: list[str], *, as_json: bool) -> None:
    running = _running()
    by_target: dict[str, list[str]] = {}
    ops: list[ServiceChangeOp] = []
    for name in names:
        matches = _matching(running, name)
        if len(matches) == 0:
            ops.append(delete_service_op(name, None, 'skipped'))
            continue
        if len(matches) > 1:
            addresses = ', '.join(sorted(_address(d) for d in matches))
            print(f'pxt service stop: {name!r} is ambiguous; it names {addresses}', file=sys.stderr)
            sys.exit(EXIT_ERROR)
        deployment = matches[0]
        by_target.setdefault(deployment['base_path'], []).append(deployment['name'])

    for target, target_names in by_target.items():
        ops += post_request('/api/localservice/stop', {'names': target_names, 'target': target})
    _print_ops(ops, as_json=as_json, verb='stopped')


def _list(target: str | None, *, as_json: bool) -> None:
    running = _running(target)
    if as_json:
        print(json.dumps(running, indent=2))
        return
    if len(running) == 0:
        print('no services running')
        return
    width = max(len(_address(d)) for d in running)
    for d in running:
        print(f'{_address(d):<{width}s}  {d["endpoint"]}  pid {d["pid"]}  {d["app_file"]}')


def _running(target: str | None = None) -> list[ServiceDeployment]:
    params = {} if target is None else {'target': target}
    deployments: list[ServiceDeployment] = get_request('/api/localservice/list', params)
    return deployments


def _address(deployment: ServiceDeployment) -> str:
    """The service's name qualified by the directory it is bound to, as 'stop' accepts it."""
    base_path = deployment['base_path']
    return deployment['name'] if base_path == '' else f'{base_path}/{deployment["name"]}'


def _matching(running: list[ServiceDeployment], name: str) -> list[ServiceDeployment]:
    """The deployments a name denotes: an address matches one, a bare name matches every target holding it."""
    return [d for d in running if name in (_address(d), d['name'])]


def _print_plan(plan: ServicePlan, *, as_json: bool, applied: bool = False) -> None:
    if as_json:
        print(json.dumps(plan, indent=2))
        return
    for service in plan['services']:
        resolution = service['resolution']
        state = service.get('status', _PENDING[resolution]) if applied else _PENDING[resolution]
        line = f'{_MARKERS[resolution]} {service["name"]:<24s} {state}'
        if service['endpoint'] is not None:
            line += f'  {service["endpoint"]}'
        print(line)
        for op in service['ops']:
            print(f'    {op["description"]}  [{op["severity"]}]')
        if service['route_detail'] is not None and resolution in ('unsupported', 'blocked'):
            print(f'    {service["route_detail"]}')
    for name in plan['extras']:
        print(f'! {name:<24s} extra (not declared); stop it with prune')

    s = plan['summary']
    updates = s['update_additive'] + s['update_destructive']
    counts = f'{s["create"]} start, {updates} restart, {s["up_to_date"]} unchanged, {s["extras"]} extra'
    if s['blocked'] > 0:
        counts += f', {s["blocked"]} blocked'
    if s['unsupported'] > 0:
        counts += f', {s["unsupported"]} unsupported'
    print()
    print(f'Plan: {counts}')


def _print_ops(ops: list[ServiceChangeOp], *, as_json: bool, verb: str) -> None:
    if as_json:
        print(json.dumps(ops, indent=2))
        return
    for op in ops:
        endpoint = op['details'].get('endpoint', '')
        suffix = f'  {endpoint}' if endpoint != '' else ''
        print(f'{op["name"]:<24s} {op.get("status", verb)}{suffix}')
    print()
    print(f'{verb}: {sum(1 for op in ops if op.get("status") == "applied")} service(s)')

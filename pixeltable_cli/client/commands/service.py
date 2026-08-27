"""`pxt service {diff,update,prune,stop,list,example}` - run the services an application file declares."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...service_types import ServiceChangeOp, ServiceInstance, ServicePlan, ServiceResolution, delete_service_op
from ...utils import PxtPath
from ..confirm import confirm_or_exit
from ..parser import Parser
from ..utils import check_file, get_request, post_request

_EXAMPLE_APP = '''\
"""Pixeltable application, written by 'pxt service example'.

One file holds both: the models, which name the tables, and the services, which serve routes over them.
The target given on the command line says which catalog directory those tables live in, so the same file
can be applied to a development directory and a production one.

    pxt schema update app.py TARGET        # create the tables the models declare
    pxt service update app.py TARGET       # serve this file's services against them

A udf defined here is referenced by this file's path, so moving or renaming the file leaves the columns that
call it unable to compute.
"""

from __future__ import annotations  # required to declare a model on Python 3.14+

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


# a udf: a Python function the computed columns below can call
@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    doc_id: pxt.Int
    title: pxt.String
    body: pxt.String | None
    title_upper = pxtf.string.upper(title)  # a computed column: an assignment, not an annotation
    summary = excerpt(title)  # a computed column over a udf this file defines


# the router names the service; without name= it takes the name of the variable holding it
ingest = FastAPIRouter(name='ingest')

# POST /docs inserts a row and returns the computed column
ingest.add_insert_route(
    Docs, path='/docs', inputs=[Docs.doc_id, Docs.title, Docs.body], outputs=[Docs.title_upper, Docs.summary]
)

# POST /titles computes without storing a row
ingest.add_compute_route(Docs, path='/titles', inputs=[Docs.title], outputs=[Docs.title_upper])
'''

_APP_FILE = """
Project:
  APP has to sit under a project root: the directory holding the project configuration, which is a
  pixeltable.toml or a pyproject.toml with a [tool.pixeltable] section. 'pxt init' writes one.
  Every local module path is relative to that root, so a udf this file defines is recorded as
  <path from the root>.<name> -- which is how the serving process finds it again."""

DIFF_EPILOG = f"""\
Examples:
  pxt service diff app.py my_dir          # what update would change; exit 2 if anything is pending
  pxt service diff app.py my_dir --json
  pxt service diff app.py my_dir --otel     # also report tracing that is off but was asked for

Tracing:
  --otel emits OpenTelemetry traces from the services 'update' starts, and needs the instrumentation
  package ('pip install pixeltable[otel]'). The setting belongs to the running service, not to the file:
  a service already running without it restarts when 'update' is given the flag, and 'diff' reports that
  as a pending change.
{_APP_FILE}"""

UPDATE_EPILOG = f"""\
Examples:
  pxt service update app.py my_dir                       # start what is declared, restart what changed
  pxt service update app.py my_dir --allow-destructive   # also stop serving routes that changed or went away
  pxt service update app.py my_dir --otel                # emit OpenTelemetry traces from what it starts

Tracing:
  --otel needs the instrumentation package ('pip install pixeltable[otel]'). The setting belongs to the
  running service, not to the file: a service already running without it restarts to pick it up, and
  dropping the flag restarts it again.
{_APP_FILE}"""

RUN_EPILOG = f"""\
Examples:
  pxt service run app.py my_dir              # the only service the file declares, until interrupted
  pxt service run app.py my_dir ingest       # the named one, when the file declares several
  pxt service run app.py my_dir --port 9000
  pxt service run app.py my_dir --otel       # emit OpenTelemetry traces from this process

One service per process, as 'update' deploys them. Nothing is recorded: it runs for as long as this process
does. Use 'update' to run it in the background, where 'list' and 'stop' can find it again.
{_APP_FILE}"""

PRUNE_EPILOG = f"""\
Examples:
  pxt service prune app.py my_dir     # stop and forget the services the file does not declare

A stopped service can be started again with 'pxt service update'.
{_APP_FILE}"""

STOP_EPILOG = """\
Examples:
  pxt service stop ingest             # a bare name, when only one target has a service of that name
  pxt service stop my_dir/ingest      # the service of that name under my_dir
  pxt service stop ingest reader
"""

LIST_EPILOG = """\
Examples:
  pxt service list                    # every service running locally
  pxt service list my_dir             # those bound at my_dir and below it
"""

CHECK_EPILOG = f"""\
Examples:
  pxt service check app.py                 # before deploying it anywhere
  pxt service check app.py --json

Exit codes:
  0  the file is valid; warnings may still be printed
  1  error: bad arguments, the file failed to import, or a udf it records cannot be read back

Notes:
  Checks what the file says on its own: it imports without modifying the catalog, it declares a
  service and a model base, and every udf its columns call is named by a module path another
  process resolves. Takes no TARGET, so it says nothing about what a target can serve;
  'pxt service diff' answers that.
{_APP_FILE}"""

VERBS = ('diff', 'update', 'run', 'prune', 'stop', 'list', 'check', 'example')

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
            '  run      serve one of them from this process instead, until interrupted\n'
            '  prune    stop and forget the services at TARGET that APP does not declare\n'
            '  stop     stop the named services\n'
            '  list     what is running locally, and where\n'
            '  check    validate the application file on its own (takes no TARGET)\n'
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

    if verb == 'check':
        ap = Parser(prog='pxt service check', epilog=CHECK_EPILOG, usage_exit_code=EXIT_ERROR)
        ap.add_argument('app', help='path to a Python file declaring FastAPIRouter services')
        ap.add_argument('--json', action='store_true', dest='as_json')
        args = ap.parse_args(argv[1:])
        check_file('/api/service/check', 'app_file', args.app, verb='service check', as_json=args.as_json)
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

    epilogs = {'diff': DIFF_EPILOG, 'update': UPDATE_EPILOG, 'run': RUN_EPILOG, 'prune': PRUNE_EPILOG}
    # a usage error exits EXIT_ERROR, not argparse's 2, which here means that changes are pending
    ap = Parser(prog=f'pxt service {verb}', epilog=epilogs[verb], usage_exit_code=EXIT_ERROR)
    ap.add_argument('app', help='path to a Python file declaring services')
    ap.add_argument('target', help='catalog directory the services bind against')
    ap.add_argument('--json', action='store_true', dest='as_json')
    if verb in ('update', 'prune'):
        ap.add_argument('-f', '--force', action='store_true', help='skip confirmation')
        ap.add_argument('-n', '--dry-run', action='store_true', dest='dry_run')
    if verb == 'diff':
        ap.add_argument('--otel', action='store_true', help='compare the running services against tracing being on')
    if verb == 'update':
        ap.add_argument('--otel', action='store_true', help='emit OpenTelemetry traces (requires `pixeltable[otel]`)')
        ap.add_argument(
            '--allow-destructive',
            action='store_true',
            dest='allow_destructive',
            help='permit changes that stop serving a route callers may be using',
        )
    if verb == 'run':
        ap.add_argument(
            'service', nargs='?', help='the service to serve; required when the file declares more than one'
        )
        ap.add_argument('--host', default='127.0.0.1', help='bind address (default: 127.0.0.1)')
        ap.add_argument('--port', type=int, default=8000, help='bind port (default: 8000)')
        ap.add_argument('--otel', action='store_true', help='emit OpenTelemetry traces (requires `pixeltable[otel]`)')
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
        _diff(app_file, args.target, as_json=args.as_json, otel=args.otel)
    elif verb == 'prune':
        _prune(app_file, args.target, as_json=args.as_json, force=args.force, dry_run=args.dry_run)
    elif verb == 'run':
        _run_foreground(
            app_file,
            args.target,
            service_name=args.service,
            host=args.host,
            port=args.port,
            as_json=args.as_json,
            otel=args.otel,
        )
    else:
        _update(
            app_file,
            args.target,
            as_json=args.as_json,
            force=args.force,
            dry_run=args.dry_run,
            allow_destructive=args.allow_destructive,
            otel=args.otel,
        )


def _example(out: str | None) -> None:
    if out is None:
        sys.stdout.write(_EXAMPLE_APP)
        return
    Path(out).write_text(_EXAMPLE_APP, encoding='utf-8')
    print(f'wrote {out}')


def _service_plan(app_file: str, target: PxtPath, otel: bool = False) -> ServicePlan:
    plan: ServicePlan = post_request('/api/service/diff', {'app_file': app_file, 'target': target, 'otel': otel})
    return plan


def _diff(app_file: str, target: PxtPath, *, as_json: bool, otel: bool = False) -> None:
    plan = _service_plan(app_file, target, otel)
    _print_plan(plan, as_json=as_json)
    sys.exit(EXIT_IN_AGREEMENT if plan['in_agreement'] else EXIT_CHANGES_PENDING)


def _update(
    app_file: str,
    target: PxtPath,
    *,
    as_json: bool,
    force: bool,
    dry_run: bool,
    allow_destructive: bool,
    otel: bool = False,
) -> None:
    plan = _service_plan(app_file, target, otel)
    if plan['in_agreement']:
        # report the same shape as a run that applied something, so a caller reading --json sees one form
        for service in plan['services']:
            service['status'] = 'skipped'
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
        '/api/service/update',
        {'app_file': app_file, 'target': target, 'allow_destructive': allow_destructive, 'otel': otel},
    )
    _print_plan(applied, as_json=as_json, applied=True)


def _run_foreground(
    app_file: str, target: PxtPath, *, service_name: str | None, host: str, port: int, as_json: bool, otel: bool
) -> None:
    """Serve one of the file's services from this process, on one port, until interrupted.

    One service per process, as `update` deploys them, so what runs here serves the same routes at the same
    paths. Nothing is recorded and nothing is reconciled: the service runs for as long as this process
    does. That is what makes it the mode for a container entrypoint or a dev loop.
    """
    # this command runs the server itself, so unlike the rest of the client it needs pixeltable in-process
    import uvicorn

    from pixeltable.serving._app import (
        create_app_for_services,
        init_instrumentation,
        instrument_app,
        load_service_routers,
    )

    if otel:
        # before the first Pixeltable operation, so that loading the file is traced too
        init_instrumentation()
    services = load_service_routers(app_file)
    if service_name is None:
        if len(services) > 1:
            declared = ', '.join(sorted(services))
            print(
                f'pxt service run: {app_file} declares more than one service: {declared}\nname the one to serve',
                file=sys.stderr,
            )
            sys.exit(EXIT_ERROR)
        service_name = next(iter(services))
    app = create_app_for_services(services, app_file=app_file, base_path=target, service_name=service_name)
    if otel:
        instrument_app(app)
    n_routes = len(app.routes)
    display_host = 'localhost' if host in ('0.0.0.0', '::') else host
    url = f'http://{display_host}:{port}'
    if as_json:
        print(json.dumps({'status': 'started', 'host': host, 'port': port, 'url': url, 'routes': n_routes}))
    else:
        print(f'Pixeltable is running on {url}\n  Routes: {n_routes}\n  API docs at {url}/docs')
    uvicorn.run(app, host=host, port=port, log_config=None)


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
    pruned: ServicePlan = post_request('/api/service/prune', {'app_file': app_file, 'target': target})
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
        service = matches[0]
        by_target.setdefault(service['base_path'], []).append(service['name'])

    for target, target_names in by_target.items():
        ops += post_request('/api/service/stop', {'names': target_names, 'target': target})
    _print_ops(ops, as_json=as_json, verb='stopped')


def _list(target: str | None, *, as_json: bool) -> None:
    running = _running(target)
    if target is not None and len(running) == 0:
        # nothing is bound at that directory, so the argument names one service rather than a directory:
        # a single service is inspected the way `describe` inspects one table
        running = _matching(_running(), target)
    if as_json:
        print(json.dumps(running, indent=2))
        return
    if len(running) == 0:
        print('no services running')
        return
    width = max(len(_address(d)) for d in running)
    for d in running:
        pid_or_state = f'pid {d["pid"]}' if d['pid'] is not None else d['state']
        # shown as the file it names: a catalog path never carries a .py suffix
        app_file = d['app_module'].replace('.', '/') + '.py'
        print(f'{_address(d):<{width}s}  {d["endpoint"]}  {pid_or_state}  {app_file}')
        prefix = d['spec']['prefix']
        for route in d['spec']['routes']:
            served = ', '.join(route['outputs']) if len(route['outputs']) > 0 else '-'
            accepted = ', '.join([*route['inputs'], *(f'{n} (file)' for n in route['uploadfile_inputs'])])
            print(
                f'    {route["method"]:<5s} {prefix}{route["path"]:<24s} {route["route_type"]:<8s} '
                f'in: {accepted or "-"}  out: {served}'
            )


def _running(target: str | None = None) -> list[ServiceInstance]:
    params = {} if target is None else {'target': target}
    instances: list[ServiceInstance] = get_request('/api/service/list', params)
    return instances


def _address(service: ServiceInstance) -> str:
    """The service's name qualified by the directory it is bound to, as 'stop' accepts it."""
    base_path = service['base_path']
    return service['name'] if base_path == '' else f'{base_path}/{service["name"]}'


def _matching(running: list[ServiceInstance], name: str) -> list[ServiceInstance]:
    """The services a name denotes: an address matches one, a bare name matches every target holding it."""
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

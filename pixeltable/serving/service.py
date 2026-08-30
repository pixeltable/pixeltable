from __future__ import annotations

import dataclasses
from typing import Literal

from pixeltable import catalog, exceptions as excs
from pixeltable.config import Config
from pixeltable.service.db import published_fingerprint
from pixeltable.utils.app_module import (
    check_report,
    get_model_bases,
    get_module_services,
    load_app_module,
    model_mismatch_error_str,
    module_routers,
    service_spec,
    services_by_name,
    visible_models,
)
from pixeltable.utils.project import ProjectFingerprint, Scope, loaded_fingerprint
from pixeltable_cli.types import (
    CheckReport,
    Resolution,
    RouteComparison,
    RouteSpec,
    ServiceChangeOp,
    ServiceDiff,
    ServiceInstance,
    ServicePlan,
    ServiceSpec,
)
from pixeltable_cli.utils import PxtPath

from . import service_instance
from .service_manager import get_manager

_DESTRUCTIVE_HINT = "Re-run 'pxt service update' with --allow-destructive to apply these changes."


def service_diff(app_file: str, target: PxtPath, *, otel: bool = False) -> ServicePlan:
    """The plan to reconcile the services at target with what's in app_file."""
    app_info = _get_app_info(app_file, target)
    # one listing for every service: a proxied manager answers each get() with a round trip
    manager = get_manager(target)
    running = {i.service_name: i for i in manager.list(target)}
    return ServicePlan(
        app_file=app_file,
        target=target,
        services=[
            _service_diff(name, service, running.get(name), app_info, target, otel)
            for name, service in sorted(app_info.services.items())
        ],
        # extras are excluded from in_agreement: update never removes a service, which is what prune is for
        extras=sorted(name for name in running if name not in app_info.services),
    )


def service_update(
    app_file: str, target: PxtPath, *, allow_destructive: bool = False, otel: bool = False
) -> ServicePlan:
    """Reconcile the services at target with what's in app_file.

    Starts a service that is not running, and restarts one whose declaration changed, since a service binds
    its models once per process. A service the file does not declare is left alone, which is what prune is
    for.

    Returns the plan that was applied, each service annotated with its status: 'applied' for one that was
    started or restarted, 'skipped' for one already serving its declaration, 'refused' for one whose routes
    the database cannot serve or whose application object Pixeltable did not declare.

    Args:
        app_file: the application file declaring the services.
        target: the catalog directory the services' models bind against.
        allow_destructive: whether to apply changes that stop serving a route contract a caller may be using.
    """
    manager = get_manager(target)
    plan = service_diff(app_file, target)
    destructive = [d.name for d in plan.services if d.resolution == 'update_destructive']
    if len(destructive) > 0 and not allow_destructive:
        names = ', '.join(repr(name) for name in destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {names} would stop serving a route that callers may be using.\n{_DESTRUCTIVE_HINT}',
        )

    running = {i.service_name: i for i in manager.list(target)}
    for diff in plan.services:
        if diff.resolution == 'blocked':
            diff.status = 'refused'
            for op in diff.ops:
                op.status = 'skipped'
            continue
        if diff.resolution == 'up_to_date':
            diff.status = 'skipped'
            continue
        if diff.name in running:
            # the running service serves the old declaration; binding happens once per process, so it is replaced
            running[diff.name].stop()
        started = manager.start(app_file, diff.name, target, otel=otel)
        diff.status = 'applied'
        diff.state = started.state
        diff.endpoint = started.endpoint
        diff.exists = True
        for op in diff.ops:
            op.status = 'skipped' if op.severity == 'blocked' else 'applied'
    return plan


def service_prune(app_file: str, target: PxtPath) -> ServicePlan:
    """Stop and forget the services at target that app_file does not declare.

    A stopped service can be started again, so this is not destructive the way dropping a table is.

    Returns the plan, with one drop operation per service stopped.
    """
    declared = services_by_name(load_app_module(app_file, subject='application file'), app_file)
    extras = sorted(i.service_name for i in get_manager(target).list(target) if i.service_name not in declared)
    return ServicePlan(
        app_file=app_file, target=target, extras=extras, ops=_forget_services(extras, target, delete=True)
    )


def service_stop(names: list[str], target: PxtPath) -> list[ServiceChangeOp]:
    """Stop the instances of the named services at target and forget them.

    A name with no instance there yields a 'skipped' operation rather than an error, so that stopping a
    set of services is idempotent.
    """
    return _forget_services(names, target, delete=False)


def service_list(target: PxtPath | None = None) -> list[ServiceInstance]:
    """The instances running locally: those serving target and below it, or all of them if target is None."""
    base_path = '' if target is None else target
    instances = get_manager(base_path).list(base_path, recursive=True)
    return [i.record.to_cli_instance() for i in sorted(instances, key=lambda i: (i.base_path, i.service_name))]


def service_check(app_file: str) -> CheckReport:
    """What checking an application file on its own reports: whether it is valid, and what to fix."""
    module = load_app_module(app_file, subject='application file')
    get_module_services(module, app_file)  # refuses a file whose service declarations are invalid
    return check_report(app_file, get_model_bases(module))


@dataclasses.dataclass(frozen=True)
class _ServiceInfo:
    """One service of an application file."""

    spec: ServiceSpec

    # 'declarative' when the service is a FastAPIRouter, 'custom' when the file supplies its own application
    kind: Literal['declarative', 'custom']


@dataclasses.dataclass(frozen=True)
class _AppInfo:
    """What an application file holds, resolved against a target."""

    app_file: str
    services: dict[str, _ServiceInfo]

    # why the target cannot serve the models these services name, or None if it can
    model_mismatch_reason: str | None

    # pxt://org:db of the target's database, without the catalog path below it; empty for a local target
    db_uri: str

    # the project db_uri was last given; None for a local target, which serves the project files in place
    published: ProjectFingerprint | None

    # the project files this application imported; a running instance records the same, and a difference
    # between them is what restarts it
    fingerprint: ProjectFingerprint


def _get_app_info(app_file: str, target: PxtPath) -> _AppInfo:
    """Load app_file and report what it holds, resolved against target."""
    from pixeltable.serving import FastAPIRouter

    module = load_app_module(app_file, subject='application file')
    routers = module_routers(module)
    # serving binds every model the file reaches, so the same set has to describe the tables at the target,
    # whichever service is being reconciled
    needed = visible_models(module)
    for router in routers:
        needed.update(router.route_models())
    project_root = Config.get().project_root
    assert project_root is not None  # load_app_module() refuses a file outside a project root
    catalog_path = catalog.Path.parse(target, allow_empty_path=True)
    db_config = Config.get().get_database_config(catalog_path)
    return _AppInfo(
        app_file=app_file,
        services={
            name: _ServiceInfo(
                spec=service_spec(name, service, routers),
                kind='declarative' if isinstance(service, FastAPIRouter) else 'custom',
            )
            for name, service in services_by_name(module, app_file).items()
        },
        model_mismatch_reason=model_mismatch_error_str(needed, target),
        db_uri=catalog_path.uri_str,
        published=published_fingerprint(catalog_path),
        fingerprint=loaded_fingerprint(project_root, db_config),
    )


def _service_diff(
    name: str,
    service: _ServiceInfo,
    running: service_instance.ServiceInstance | None,
    app_info: _AppInfo,
    target: PxtPath,
    otel: bool,
) -> ServiceDiff:
    """How the instance of one service at target differs from what the application file holds."""
    ops: list[ServiceChangeOp] = []
    route_detail: str | None = None

    if running is None:
        route_comparison: RouteComparison = 'unavailable'
        route_detail = 'the service is not running at this target'
    else:
        # a declarative service is compared by its route declarations, an application object by the paths it
        # serves itself
        route_comparison = 'declarative' if service.kind == 'declarative' else 'openapi'
        ops += compare_specs(running.spec, service.spec)
        if running.otel != otel:
            ops.append(ServiceChangeOp.otel(running.otel, otel))

    published = app_info.published
    if published is not None and app_info.fingerprint.restart_needed(published):
        # a hosted service reads the project the database was given, so restarting it re-runs the old files
        scope: Scope = 'image' if app_info.fingerprint.image_needed(published) else 'restart'
        ops.append(
            ServiceChangeOp.project_moved(
                app_info.fingerprint.changes(published, scope), command=f'pxt db update {app_info.db_uri}'
            )
        )
    elif len(ops) == 0 and running is not None and app_info.fingerprint.restart_needed(running.record.fingerprint):
        # the contract is unchanged, so the project is the only thing left to report
        ops.append(ServiceChangeOp.project_moved(app_info.fingerprint.changes(running.record.fingerprint)))

    if app_info.model_mismatch_reason is not None:
        command = f'pxt schema update {app_info.app_file}' + ('' if target == '' else f' {target}')
        ops.append(ServiceChangeOp.blocked_schema(name, app_info.model_mismatch_reason, command))

    resolution: Resolution
    if any(op.severity == 'blocked' for op in ops):
        # the database has to change before this service can serve, whether it is running yet or not
        resolution = 'blocked'
    elif running is None:
        resolution = 'create'
    elif any(op.destructive for op in ops):
        resolution = 'update_destructive'
    elif len(ops) > 0:
        resolution = 'update_additive'
    else:
        resolution = 'up_to_date'

    return ServiceDiff(
        name=name,
        exists=running is not None,
        state=None if running is None else running.state,
        endpoint=None if running is None else running.endpoint,
        base_path=target,
        kind=service.kind,
        resolution=resolution,
        route_comparison=route_comparison,
        route_detail=route_detail,
        ops=ops,
    )


def _forget_services(names: list[str], target: PxtPath, *, delete: bool) -> list[ServiceChangeOp]:
    """Stop each named instance and report one operation per name; delete also forgets the registration."""
    manager = get_manager(target)
    ops: list[ServiceChangeOp] = []
    for name in names:
        running = manager.get(name, target)
        if running is None:
            # it stopped between the listing and here; nothing to stop, and it is already forgotten
            ops.append(ServiceChangeOp.delete_service(name, None, 'skipped'))
            continue
        if delete:
            running.delete()
        else:
            running.stop()
        ops.append(ServiceChangeOp.delete_service(name, running.endpoint, 'applied'))
    return ops


def compare_specs(current: ServiceSpec, declared: ServiceSpec) -> list[ServiceChangeOp]:
    """The operations that would bring the current service definition to the declared one.

    Two routes with the same method and path serve the same callers, so a difference between them is an
    alteration of one route rather than a drop and an add. Only adding a route leaves what is already served
    untouched; every other operation replaces a contract that callers may be relying on.
    """
    ops: list[ServiceChangeOp] = []
    ops += _path_ops(current.app_paths, declared.app_paths)

    prefix_op = _prefix_change(current, declared)
    if prefix_op is not None:
        # every route moved, so the routes below would each read as a drop and an add of the same contract
        return [*ops, prefix_op]

    # keyed by method and path, which ignores declaration order: valid paths don't contain parameters, which avoids
    # ambiguity
    current_routes = {(r.method, r.path): r for r in current.routes}
    declared_routes = {(r.method, r.path): r for r in declared.routes}

    for key, route in declared_routes.items():
        name = _route_name(route)
        previous = current_routes.get(key)
        if previous is None:
            ops.append(ServiceChangeOp.route_added(name))
            continue
        changed = _changed_fields(previous, route)
        if len(changed) > 0:
            ops.append(ServiceChangeOp.route_replaced(name, changed))

    for key, route in current_routes.items():
        if key in declared_routes:
            continue
        name = _route_name(route)
        ops.append(ServiceChangeOp.route_dropped(name))

    return ops


def _prefix_change(current: ServiceSpec, declared: ServiceSpec) -> ServiceChangeOp | None:
    """The one operation that accounts for every path difference, when a shared prefix moved.

    A prefix change breaks every caller at once, so it reads better as one operation than as a drop and an
    add of every route. It applies where stripping each side's shared prefix leaves the same routes.
    """
    if len(current.routes) == 0 or len(declared.routes) == 0:
        return None
    current_prefix = _common_prefix([route.path for route in current.routes])
    declared_prefix = _common_prefix([route.path for route in declared.routes])
    if current_prefix == declared_prefix:
        return None
    if _without_prefix(current.routes, current_prefix) != _without_prefix(declared.routes, declared_prefix):
        return None
    return ServiceChangeOp.prefix_moved(declared.name, current_prefix, declared_prefix)


def _path_ops(current: list[str], declared: list[str]) -> list[ServiceChangeOp]:
    """The operations that would bring the served paths to the declared ones."""
    ops: list[ServiceChangeOp] = []
    ops += [ServiceChangeOp.path_added(path) for path in sorted(set(declared) - set(current))]
    ops += [ServiceChangeOp.path_dropped(path) for path in sorted(set(current) - set(declared))]
    return ops


def _route_name(route: RouteSpec) -> str:
    """The route as it reads in a diff, eg 'POST /v1/ingest'."""
    return f'{route.method} {route.path}'


def _common_prefix(paths: list[str]) -> str:
    """The leading segments shared by every path's parent."""
    segments = [path.rsplit('/', 1)[0].split('/') for path in paths]
    shared: list[str] = []
    for parts in zip(*segments):
        if len(set(parts)) > 1:
            break
        shared.append(parts[0])
    return '/'.join(shared)


def _without_prefix(routes: list[RouteSpec], prefix: str) -> set[tuple[str, str]]:
    return {(route.method, route.path[len(prefix) :]) for route in routes}


def _changed_fields(current: RouteSpec, declared: RouteSpec) -> list[str]:
    """The fields in which two route declarations differ."""
    return sorted(name for name in type(declared).model_fields if getattr(current, name) != getattr(declared, name))

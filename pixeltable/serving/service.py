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
from pixeltable.utils.project import ProjectFingerprint, loaded_fingerprint
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


def _base_path(target: PxtPath) -> str:
    """The catalog directory inside target's database, as an instance records it."""
    return '/'.join(catalog.Path.parse(target, allow_empty_path=True).components)


def service_diff(app_file: str, target: PxtPath, *, otel: bool = False) -> ServicePlan:
    """The plan to reconcile the services at target with what's in app_file."""
    app_info = _get_app_info(app_file, target)
    # one listing for every service: a proxied manager answers each get() with a round trip
    manager = get_manager(target)
    running = {i.service_name: i for i in manager.list(_base_path(target))}
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
        allow_destructive: whether to apply changes that stop serving a route callers may be using.
    """
    manager = get_manager(target)
    plan = service_diff(app_file, target, otel=otel)
    destructive = [d.name for d in plan.services if d.resolution == 'update_destructive']
    if len(destructive) > 0 and not allow_destructive:
        names = ', '.join(repr(name) for name in destructive)
        raise excs.RequestError(
            excs.ErrorCode.DESTRUCTIVE_SCHEMA_CHANGE,
            f'Reconciling {names} would stop serving a route that callers may be using.\n{_DESTRUCTIVE_HINT}',
        )

    running = {i.service_name: i for i in manager.list(_base_path(target))}
    for diff in plan.services:
        if diff.resolution == 'blocked':
            diff.status = 'refused'
            for op in diff.ops:
                op.status = 'skipped'
            continue
        if diff.resolution == 'up_to_date':
            diff.status = 'skipped'
            continue
        instance = running.get(diff.name)
        if instance is not None and instance.state is service_instance.ServiceInstanceState.AVAILABLE:
            # the running service serves the old declaration; binding happens once per process, so it is replaced
            instance.stop()
        started = manager.start(app_file, diff.name, _base_path(target), otel=otel)
        diff.status = 'applied'
        diff.state = started.state
        diff.endpoint = started.endpoint
        diff.exists = True
        for op in diff.ops:
            op.status = 'skipped' if op.severity == 'blocked' else 'applied'
    return plan


def service_prune(app_file: str, target: PxtPath, *, dry_run: bool = False) -> ServicePlan:
    """Stop and forget the services that aren't in app_file. Returns the plan, with one drop operation per service. """
    declared = services_by_name(load_app_module(app_file, subject='application file'), app_file)
    extras = sorted(
        i.service_name for i in get_manager(target).list(_base_path(target)) if i.service_name not in declared
    )
    ops = (
        [ServiceChangeOp.delete_service(name, None, 'skipped') for name in extras]
        if dry_run
        else _forget_services(extras, target, delete=True)
    )
    return ServicePlan(app_file=app_file, target=target, extras=extras, ops=ops)


def service_stop(names: list[str]) -> list[ServiceChangeOp]:
    """Stop the named instances and forget them.

    A unrecognized name yields a 'skipped' operation rather than an error, so that stopping a set of
    services is idempotent.
    """
    ops: list[ServiceChangeOp] = []
    for name in names:
        found = _resolve_service_instances(name)
        if len(found) == 0:
            ops.append(ServiceChangeOp.delete_service(name, None, 'skipped'))
            continue
        if len(found) > 1:
            addresses = ', '.join(sorted(i.record.to_cli_instance().address for i in found))
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} is ambiguous; it names {addresses}')
        found[0].stop()
        ops.append(ServiceChangeOp.delete_service(name, found[0].endpoint, 'applied'))
    return ops


def _resolve_service_instances(name_or_uri: str) -> list[service_instance.ServiceInstance]:
    """Return the instances matching a service uri or name.

    A uri matches exactly one instance; a bare service name matches every locally running instance with that same name.
    """
    path = catalog.Path.parse(name_or_uri, allow_empty_path=True)
    if path.len > 1 or not path.is_local:
        target = PxtPath(str(path.parent))
        found = get_manager(target).get(path.name, _base_path(target))
        return [] if found is None else [found]
    return [i for i in get_manager('').list('', recursive=True) if i.service_name == name_or_uri]


def service_list(target: PxtPath | None = None) -> list[ServiceInstance]:
    """The instances serving target and below it, or every local one when target is None.

    A target holding no instance is taken to name one service, which is then the only one reported: a
    service is inspected the way `describe` inspects one table.
    """
    catalog_uri = PxtPath('') if target is None else target
    instances = get_manager(catalog_uri).list(_base_path(catalog_uri), recursive=True)
    if len(instances) == 0 and target is not None:
        instances = _resolve_service_instances(target)
    uri = catalog.Path.parse(catalog_uri, allow_empty_path=True).uri_str
    return [i.record.to_cli_instance(uri) for i in sorted(instances, key=lambda i: (i.base_path, i.service_name))]


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
    """How the instance of one service at target differs from what's reflected in app_info."""
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
    # empty for a local target, whose services read the project files in place
    unpublished = app_info.fingerprint.compare(published, own_files_only=True) if published is not None else set()
    if app_info.db_uri != '' and published is None:
        # pxt db update creates the database and gives it both artifacts; until it runs there is nothing here
        ops.append(ServiceChangeOp.db_not_updated(f'pxt db update {app_info.db_uri}'))
    elif len(unpublished) > 0:
        # this requires a pxt db update
        ops.append(
            ServiceChangeOp.project_moved(
                app_info.fingerprint.changes(published, unpublished, own_files_only=True),
                command=f'pxt db update {app_info.db_uri}',
            )
        )
    elif len(ops) == 0 and running is not None:
        stale = app_info.fingerprint.compare(running.record.fingerprint)
        if len(stale) > 0:
            # the routes agree, but some archive files changed
            ops.append(ServiceChangeOp.project_moved(app_info.fingerprint.changes(running.record.fingerprint, stale)))

    if app_info.model_mismatch_reason is not None:
        command = f'pxt schema update {app_info.app_file}' + ('' if target == '' else f' {target}')
        ops.append(ServiceChangeOp.blocked_schema(name, app_info.model_mismatch_reason, command))

    resolution: Resolution
    if any(op.severity == 'blocked' for op in ops):
        # the database has to change before this service can serve, whether it is running yet or not
        resolution = 'blocked'
    elif running is None:
        resolution = 'create'
    elif running.state is not service_instance.ServiceInstanceState.AVAILABLE:
        # registered but not serving, whatever its declaration says: an update starts it
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
        running = manager.get(name, _base_path(target))
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
    untouched; every other operation changes a route callers may be relying on.
    """
    ops: list[ServiceChangeOp] = []
    ops += _path_ops(current.app_paths, declared.app_paths)

    prefix_op = _prefix_change(current, declared)
    if prefix_op is not None:
        # every route moved, so the routes below would each read as a drop and an add of the same route
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

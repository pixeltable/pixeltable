"""The operations that reconcile a running service with the definition an application file holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pixeltable import catalog
from pixeltable.config import Config
from pixeltable.utils.app_module import (
    load_app_module,
    model_mismatch_error_str,
    module_routers,
    service_spec,
    services_by_name,
    visible_models,
)
from pixeltable.utils.project import ProjectFingerprint, loaded_fingerprint
from pixeltable_cli.types import (
    Resolution,
    RouteComparison,
    RouteSpec,
    ServiceChangeOp,
    ServiceDiff,
    ServicePlan,
    ServiceSpec,
)
from pixeltable_cli.utils import PxtPath

if TYPE_CHECKING:
    import fastapi

    from pixeltable.serving import FastAPIRouter
    from pixeltable.serving.service_manager import ServiceManagerBase


def service_diff(app_file: str, target: PxtPath, *, otel: bool = False) -> ServicePlan:
    """The changes that reconciling the instances at target with the services app_file declares would make.

    Read-only: nothing is started, stopped or forgotten.

    Args:
        app_file: the application file declaring the services.
        target: the catalog directory the services' models bind against.
    """
    from pixeltable.serving.service_manager import get_manager

    manager = get_manager(target)
    module = load_app_module(app_file, subject='application file')
    routers = module_routers(module)
    # serving binds every model the file reaches, so the same set has to describe the tables at the target,
    # whichever service is being reconciled
    needed = visible_models(module)
    for router in routers:
        needed.update(router.route_models())
    mismatch = model_mismatch_error_str(needed, target)
    project_root = Config.get().project_root
    assert project_root is not None  # load_app_module() refuses a file outside a project root
    db_config = Config.get().get_database_config(catalog.Path.parse(target, allow_empty_path=True))
    fingerprint = loaded_fingerprint(project_root, db_config)
    diffs = [
        _service_diff(
            manager, name, service, service_spec(name, service, routers), mismatch, fingerprint, app_file, target, otel
        )
        for name, service in sorted(services_by_name(module, app_file).items())
    ]
    return _plan_from_service_diffs(manager, diffs, app_file, target)


def _service_diff(
    manager: ServiceManagerBase,
    name: str,
    service: FastAPIRouter | fastapi.FastAPI,
    declared_spec: ServiceSpec,
    model_mismatch_reason: str | None,
    fingerprint: ProjectFingerprint,
    app_file: str,
    target: PxtPath,
    otel: bool = False,
) -> ServiceDiff:
    """How the instance of one declared service at target differs from its declaration."""
    from pixeltable.serving import FastAPIRouter

    running = manager.get(name, target)
    ops: list[ServiceChangeOp] = []
    route_detail: str | None = None
    declared = service if isinstance(service, FastAPIRouter) else None
    kind: Literal['declarative', 'custom'] = 'custom' if declared is None else 'declarative'

    if running is None:
        route_comparison: RouteComparison = 'unavailable'
        route_detail = 'the service is not running at this target'
    else:
        # a declared service is compared by its route declarations, an application object by the paths it
        # serves itself
        route_comparison = 'declarative' if declared is not None else 'openapi'
        ops += compare_specs(running.spec, declared_spec)
        if running.otel != otel:
            ops.append(ServiceChangeOp.otel(running.otel, otel))

    recorded = None if running is None else running.record.fingerprint
    if len(ops) == 0 and recorded is not None and fingerprint.restart_needed(recorded):
        # the contract is unchanged, so the project is the only thing left to report
        ops.append(ServiceChangeOp.project_moved(fingerprint.changes(recorded)))

    if model_mismatch_reason is not None:
        command = f'pxt schema update {app_file}' + ('' if target == '' else f' {target}')
        ops.append(ServiceChangeOp.blocked_schema(name, model_mismatch_reason, command))

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
        kind=kind,
        resolution=resolution,
        route_comparison=route_comparison,
        route_detail=route_detail,
        ops=ops,
    )


def _plan_from_service_diffs(
    manager: ServiceManagerBase, diffs: list[ServiceDiff], app_file: str, target: PxtPath
) -> ServicePlan:
    """The plan that the given per-service diffs describe."""
    declared = {diff.name for diff in diffs}
    return ServicePlan(
        app_file=app_file,
        target=target,
        services=diffs,
        # extras are excluded from in_agreement: update never removes a service, which is what prune is for
        extras=sorted(i.service_name for i in manager.list(target) if i.service_name not in declared),
    )


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


def _without_prefix(routes: list[RouteSpec], prefix: str) -> set[tuple[str, str]]:
    return {(route.method, route.path[len(prefix) :]) for route in routes}


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


def _path_ops(current: list[str], declared: list[str]) -> list[ServiceChangeOp]:
    """The operations that would bring the served paths to the declared ones."""
    ops: list[ServiceChangeOp] = []
    ops += [ServiceChangeOp.path_added(path) for path in sorted(set(declared) - set(current))]
    ops += [ServiceChangeOp.path_dropped(path) for path in sorted(set(current) - set(declared))]
    return ops


def _changed_fields(current: RouteSpec, declared: RouteSpec) -> list[str]:
    """The fields in which two route declarations differ."""
    return sorted(name for name in type(declared).model_fields if getattr(current, name) != getattr(declared, name))

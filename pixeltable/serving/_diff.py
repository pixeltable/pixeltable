"""The operations that reconcile a running service with the definition an application file holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

if TYPE_CHECKING:
    from ._spec import RouteSpec, ServiceSpec

# Extends the severities a schema change uses with 'blocked': an operation that service update cannot carry
# out because the database, not the service, has to satisfy it. The command that does so is in details.
# Mirrored by pixeltable_cli.service_types.Severity.
Severity = Literal['additive', 'destructive', 'unsupported', 'blocked']


class ServiceChangeOp(TypedDict):
    """One operation reconciling a running service with the definition an application file holds.

    Mirrored by pixeltable_cli.service_types.ServiceChangeOp; adding, removing or retyping a field here means
    doing the same there.
    """

    target: Literal['service', 'base_path', 'route', 'resources', 'secret', 'runtime']

    # route: the method and path, eg 'POST /v1/ingest'; resources: the field; secret: the key
    name: str

    op: Literal['add', 'drop', 'alter']
    severity: Severity
    description: str  # one sentence, ready to print
    details: dict[str, str]  # 'from' and 'to' for an alter, 'command' for a blocked operation


def otel_op(current: bool, requested: bool) -> ServiceChangeOp:
    state = {True: 'on', False: 'off'}
    return {
        'target': 'service',
        'name': 'otel',
        'op': 'alter',
        'severity': 'additive',  # the routes are unchanged; the service restarts to pick up the new setting
        'description': f'tracing will be turned {state[requested]}, which restarts the service',
        'details': {'from': state[current], 'to': state[requested]},
    }


def blocked_schema_op(service_name: str, description: str, command: str) -> ServiceChangeOp:
    return {
        'target': 'service',
        'name': service_name,
        'op': 'alter',
        'severity': 'blocked',
        'description': description,
        'details': {'command': command},
    }


def _route_name(route: RouteSpec) -> str:
    """The route as it reads in a diff, eg 'POST /v1/ingest'."""
    return f'{route["method"]} {route["path"]}'


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
    if len(current['routes']) == 0 or len(declared['routes']) == 0:
        return None
    current_prefix = _common_prefix([route['path'] for route in current['routes']])
    declared_prefix = _common_prefix([route['path'] for route in declared['routes']])
    if current_prefix == declared_prefix:
        return None
    if _without_prefix(current['routes'], current_prefix) != _without_prefix(declared['routes'], declared_prefix):
        return None
    return {
        'target': 'service',
        'name': declared['name'],
        'op': 'alter',
        'severity': 'destructive',
        'description': (
            f'service {declared["name"]!r} will be served at prefix {declared_prefix!r} rather than {current_prefix!r}'
        ),
        'details': {'from': current_prefix, 'to': declared_prefix},
    }


def _without_prefix(routes: list[RouteSpec], prefix: str) -> set[tuple[str, str]]:
    return {(route['method'], route['path'][len(prefix) :]) for route in routes}


def compare_specs(current: ServiceSpec, declared: ServiceSpec) -> list[ServiceChangeOp]:
    """The operations that would bring the current service definition to the declared one.

    Two routes with the same method and path serve the same callers, so a difference between them is an
    alteration of one route rather than a drop and an add. Only adding a route leaves what is already served
    untouched; every other operation replaces a contract that callers may be relying on.
    """
    ops: list[ServiceChangeOp] = []
    ops += _path_ops(current['app_paths'], declared['app_paths'])

    prefix_op = _prefix_change(current, declared)
    if prefix_op is not None:
        # every route moved, so the routes below would each read as a drop and an add of the same contract
        return [*ops, prefix_op]

    # keyed by method and path, which ignores declaration order: valid paths don't contain parameters, which avoids
    # ambiguity
    current_routes = {(r['method'], r['path']): r for r in current['routes']}
    declared_routes = {(r['method'], r['path']): r for r in declared['routes']}

    for key, route in declared_routes.items():
        name = _route_name(route)
        previous = current_routes.get(key)
        if previous is None:
            ops.append(
                {
                    'target': 'route',
                    'name': name,
                    'op': 'add',
                    'severity': 'additive',
                    'description': f'route {name!r} will be added',
                    'details': {},
                }
            )
            continue
        changed = _changed_fields(previous, route)
        if len(changed) > 0:
            ops.append(
                {
                    'target': 'route',
                    'name': name,
                    'op': 'alter',
                    'severity': 'destructive',
                    'description': f'route {name!r} will be replaced: {", ".join(changed)} changed',
                    'details': {'changed': ', '.join(changed)},
                }
            )

    for key, route in current_routes.items():
        if key in declared_routes:
            continue
        name = _route_name(route)
        ops.append(
            {
                'target': 'route',
                'name': name,
                'op': 'drop',
                'severity': 'destructive',
                'description': f'route {name!r} will no longer be served',
                'details': {},
            }
        )

    return ops


def _path_ops(current: list[str], declared: list[str]) -> list[ServiceChangeOp]:
    """The operations that would bring the served paths to the declared ones."""
    ops: list[ServiceChangeOp] = []
    for path in sorted(set(declared) - set(current)):
        ops.append(
            {
                'target': 'route',
                'name': path,
                'op': 'add',
                'severity': 'additive',
                'description': f'path {path!r} will be served',
                'details': {},
            }
        )
    for path in sorted(set(current) - set(declared)):
        ops.append(
            {
                'target': 'route',
                'name': path,
                'op': 'drop',
                'severity': 'destructive',
                'description': f'path {path!r} will no longer be served',
                'details': {},
            }
        )
    return ops


def _changed_fields(current: RouteSpec, declared: RouteSpec) -> list[str]:
    """The fields in which two route declarations differ."""
    # cast: mypy indexes a TypedDict by literal keys only
    current_fields = cast(dict[str, Any], current)
    declared_fields = cast(dict[str, Any], declared)
    return sorted(k for k in declared_fields if current_fields[k] != declared_fields[k])

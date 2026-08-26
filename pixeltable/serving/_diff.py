"""The operations that reconcile a service deployment with the definition an application file holds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

if TYPE_CHECKING:
    from ._spec import RouteSpec, ServiceSpec

# Extends the severities a schema change uses with 'blocked': an operation that service update cannot carry
# out because the database, not the deployment, has to satisfy it. The command that does so is in details.
# Mirrored by pixeltable_cli.service_types.Severity.
Severity = Literal['additive', 'destructive', 'unsupported', 'blocked']


class ServiceChangeOp(TypedDict):
    """One operation reconciling a service deployment with the definition an application file holds.

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


def blocked_schema_op(service_name: str, description: str, command: str) -> ServiceChangeOp:
    """The operation for a service whose models do not describe the tables they name.

    A schema update is the only thing that reconciles them, so the operation carries the command that runs it.
    """
    return {
        'target': 'service',
        'name': service_name,
        'op': 'alter',
        'severity': 'blocked',
        'description': description,
        'details': {'command': command},
    }


def _route_name(route: RouteSpec, prefix: str) -> str:
    """The route as it reads in a diff, eg 'POST /v1/ingest'."""
    return f'{route["method"]} {prefix}{route["path"]}'


def compare_specs(deployed: ServiceSpec, declared: ServiceSpec) -> list[ServiceChangeOp]:
    """The operations that would bring a deployed service definition to the declared one.

    Two routes with the same method and path serve the same callers, so a difference between them is an
    alteration of one route rather than a drop and an add. Only adding a route leaves what is already served
    untouched; every other operation replaces a contract that callers may be relying on.
    """
    ops: list[ServiceChangeOp] = []

    if deployed['prefix'] != declared['prefix']:
        # the prefix is part of every route's URL, but not of the route declarations compared below
        ops.append(
            {
                'target': 'service',
                'name': declared['name'],
                'op': 'alter',
                'severity': 'destructive',
                'description': (
                    f'service {declared["name"]!r} will be served at prefix {declared["prefix"]!r} '
                    f'rather than {deployed["prefix"]!r}'
                ),
                'details': {'from': deployed['prefix'], 'to': declared['prefix']},
            }
        )

    # keyed by method and path, which ignores declaration order: valid paths don't contain parameters, which avoids
    # ambiguity
    deployed_routes = {(r['method'], r['path']): r for r in deployed['routes']}
    declared_routes = {(r['method'], r['path']): r for r in declared['routes']}

    for key, route in declared_routes.items():
        name = _route_name(route, declared['prefix'])
        previous = deployed_routes.get(key)
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

    for key, route in deployed_routes.items():
        if key in declared_routes:
            continue
        name = _route_name(route, deployed['prefix'])
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


def _changed_fields(deployed: RouteSpec, declared: RouteSpec) -> list[str]:
    """The fields in which two route declarations differ, including any either one alone has."""
    # compared as plain dicts: a record written by another version of Pixeltable can carry fields this one
    # does not declare, and dropping them from the comparison would report such a route as unchanged
    deployed_fields = cast(dict[str, Any], deployed)
    declared_fields = cast(dict[str, Any], declared)
    keys = set(deployed_fields) | set(declared_fields)
    return sorted(
        k
        for k in keys
        if k not in deployed_fields or k not in declared_fields or deployed_fields[k] != declared_fields[k]
    )

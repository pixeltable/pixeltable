"""The operations that reconcile a service deployment with the definition an application file holds."""

from __future__ import annotations

from typing import Literal, TypedDict

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


def blocked_route_op(route_name: str, description: str, base_path: str) -> ServiceChangeOp:
    """The operation for a route that the schema at `base_path` has to satisfy before the route can be served."""
    target = f' --target {base_path}' if base_path != '' else ''
    return {
        'target': 'route',
        'name': route_name,
        'op': 'alter',
        'severity': 'blocked',
        'description': description,
        'details': {'command': f'pxt schema update{target}'},
    }

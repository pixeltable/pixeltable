from typing import Any, Literal

from typing_extensions import TypedDict

# a service plan and a schema plan share these definitions; moving them to a common module is a follow-up
from pixeltable_cli.schema_types import OpStatus, _Status
from pixeltable_cli.utils import PxtPath

# Extends the severities a schema plan uses (its three, plus 'blocked'). 'blocked' marks an operation that
# 'service update' cannot carry out because the database, not the service, has to satisfy it; the command
# that does so is in the operation's details.
Severity = Literal['additive', 'destructive', 'unsupported', 'blocked']

# Extends the resolutions a schema plan uses with 'blocked': the service cannot be reconciled until the
# database satisfies what one of its routes needs, and the command that does so is in the operation's details.
ServiceResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'blocked']

# How the routes were compared. 'declarative' compares the route declarations the running service was started
# from; 'openapi' compares the OpenAPI document generated from a custom application. 'unavailable' means the
# comparison did not happen, which is not the same as happening and finding no differences.
RouteComparison = Literal['declarative', 'openapi', 'unavailable']


class ServiceChangeOp(_Status):
    """One operation reconciling a running service with the service definition an application file holds."""

    # 'secret' and 'runtime' are properties of the database a service depends on, not of the service
    # itself: operations against them are always 'blocked', never applied. There is no 'bundle' target; the
    # runtime image belongs to the database.
    target: Literal['service', 'base_path', 'route', 'resources', 'secret', 'runtime']

    # route: the method and path, eg 'POST /v1/ingest'; resources: the field, eg 'workers'; secret: the key;
    # base_path and service: the value or the service name
    name: str

    op: Literal['add', 'drop', 'alter']
    severity: Severity
    description: str  # one sentence, ready to print
    details: dict[str, str]  # 'from' and 'to' for an alter, 'command' for a blocked operation
    destructive: bool  # the boolean form of severity
    requires_restart: bool  # whether applying this interrupts the running service


class ServiceDiff(_Status):
    """How one running service differs from the definition that declares it.

    A service definition is location-independent: it names models, columns and queries, never catalog paths.
    A running service is that definition applied to a target, so name and kind describe the definition
    while base_path, state and endpoint describe the service.
    """

    name: str
    exists: bool
    state: str | None  # the service's state, None when it does not exist
    endpoint: str | None

    # the catalog path the definition's models bind against
    base_path: PxtPath

    # 'declarative' when every route comes from a route declaration, 'custom' when the file supplies its own
    # application object
    kind: Literal['declarative', 'custom']

    resolution: ServiceResolution

    route_comparison: RouteComparison
    route_detail: str | None  # why the routes were not compared, when they were not

    # empty for a create, which subsumes the additions that constitute it
    ops: list[ServiceChangeOp]

    destructive: bool  # whether any of the operations is
    requires_restart: bool


class ServicePlanSummary(TypedDict):
    up_to_date: int
    create: int
    update_additive: int
    update_destructive: int
    extras: int
    blocked: int  # services whose reconciliation the database has to enable first
    destructive: int  # operations, not services
    blocked_ops: int  # operations the database has to satisfy before the plan can be applied
    restarts: int  # services that applying the plan would interrupt


class _PlanOps(TypedDict, total=False):
    ops: list[ServiceChangeOp]  # on whole services, unlike ServiceDiff.ops


class ServicePlan(_PlanOps):
    """Set of changes needed to reconcile the services at a target with the definitions a file holds."""

    app_file: str

    # the database the services live in, and the catalog path their models bind against
    target: PxtPath

    in_agreement: bool  # True if no service needs a create or an update; extras don't count
    services: list[ServiceDiff]
    extras: list[str]  # services at the target that the file does not declare
    summary: ServicePlanSummary


class RouteSpec(TypedDict):
    """Mirror of pixeltable.serving.RouteSpec: one route of a service definition."""

    method: Literal['GET', 'POST']
    path: str  # as served, including every prefix in front of the declared path
    route_type: Literal['insert', 'update', 'delete', 'compute', 'query']

    # the model a route is declared against, or the catalog path of the table; a query route written against
    # tables has neither
    model: str | None
    table: str | None

    inputs: list[str]
    uploadfile_inputs: list[str]
    outputs: list[str]
    match_columns: list[str]

    background: bool
    return_fileresponse: bool
    one_row: bool
    export_sql: dict[str, Any] | None
    query: str | None


class ServiceSpec(TypedDict):
    """Mirror of pixeltable.serving.ServiceSpec: everything an application file declares about one service."""

    name: str
    routes: list[RouteSpec]

    # paths served by a custom FastAPI instance that don't come from FastAPIRouter instances
    app_paths: list[str]


class ServiceInstance(TypedDict):
    """A service instance; mirrors pixeltable/serving/service_instance.py."""

    name: str
    base_path: PxtPath  # the catalog directory the service's models are bound to
    endpoint: str
    state: str
    app_module: str  # dotted module path, relative to the project root
    spec: ServiceSpec  # what it serves

    pid: int | None  # the process serving the instance; set only for an instance running on this machine
    process_started_at: float | None  # creation time of pid, None where the platform does not report one


def delete_service_op(name: str, endpoint: str | None, status: OpStatus) -> ServiceChangeOp:
    """The operation for deleting the named service, in the given status."""
    served = '' if endpoint is None else f' at {endpoint}'
    return {
        'target': 'service',
        'name': name,
        'op': 'drop',
        'severity': 'destructive',
        'description': f'service {name!r}{served} will be deleted',
        'details': {} if endpoint is None else {'endpoint': endpoint},
        'destructive': True,
        'requires_restart': False,
        'status': status,
    }


__all__ = [
    'OpStatus',
    'RouteComparison',
    'RouteSpec',
    'ServiceChangeOp',
    'ServiceDiff',
    'ServiceInstance',
    'ServicePlan',
    'ServicePlanSummary',
    'ServiceResolution',
    'ServiceSpec',
    'Severity',
    'delete_service_op',
]

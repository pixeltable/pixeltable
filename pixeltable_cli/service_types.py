from typing import Literal, Optional

from typing_extensions import TypedDict

# a service plan and a schema plan share these definitions; moving them to a common module is a follow-up
from pixeltable_cli.schema_types import OpStatus, _Status
from pixeltable_cli.utils import PxtPath

# Extends the severities a schema plan uses (its three, plus 'blocked'). 'blocked' marks an operation that
# 'service update' cannot carry out because the database, not the deployment, has to satisfy it; the command
# that does so is in the operation's details.
Severity = Literal['additive', 'destructive', 'unsupported', 'blocked']

# Extends the resolutions a schema plan uses with 'blocked': the deployment cannot be reconciled until the
# database satisfies what one of its routes needs, and the command that does so is in the operation's details.
ServiceResolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported', 'blocked']

# How the routes were compared. 'declarative' compares the route declarations a deployment was created
# from; 'openapi' compares the OpenAPI document generated from a custom application. 'unavailable' means the
# comparison did not happen, which is not the same as happening and finding no differences.
RouteComparison = Literal['declarative', 'openapi', 'unavailable']


class ServiceChangeOp(_Status):
    """One operation reconciling a service deployment with the service definition an application file holds."""

    # 'secret' and 'runtime' are properties of the database a deployment depends on, not of the deployment
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
    requires_restart: bool  # whether applying this interrupts the running deployment


class ServiceDiff(_Status):
    """How one service deployment differs from the definition that declares it.

    A service definition is location-independent: it names models, columns and queries, never catalog paths.
    A deployment is that definition applied to a target, so name and kind describe the definition while
    base_path, state and endpoint describe the deployment.
    """

    name: str
    exists: bool
    state: Optional[str]  # the deployment's state, None when it does not exist
    endpoint: Optional[str]

    # the catalog path the definition's models bind against
    base_path: PxtPath

    # 'declarative' when every route comes from a route declaration, 'custom' when the file supplies its own
    # application object
    kind: Literal['declarative', 'custom']

    resolution: ServiceResolution

    route_comparison: RouteComparison
    route_detail: Optional[str]  # why the routes were not compared, when they were not

    # empty for a create, which subsumes the additions that constitute it
    ops: list[ServiceChangeOp]

    destructive: bool  # whether any of the operations is
    requires_restart: bool


class ServicePlanSummary(TypedDict):
    up_to_date: int
    create: int
    update_additive: int
    update_destructive: int
    unsupported: int
    extras: int
    blocked: int  # deployments whose reconciliation the database has to enable first
    destructive: int  # operations, not deployments
    blocked_ops: int  # operations the database has to satisfy before the plan can be applied
    restarts: int  # deployments that applying the plan would interrupt


class _PlanOps(TypedDict, total=False):
    ops: list[ServiceChangeOp]  # on whole deployments, unlike ServiceDiff.ops


class ServicePlan(_PlanOps):
    """Set of changes needed to reconcile the deployments at a target with the definitions a file holds."""

    app_file: str

    # the database the deployments live in, and the catalog path their models bind against
    target: PxtPath

    in_agreement: bool  # True if no deployment needs a create or an update; extras don't count
    services: list[ServiceDiff]
    extras: list[str]  # deployments at the target that the file does not declare
    summary: ServicePlanSummary


class ServiceDeployment(TypedDict):
    """A service running locally, as `pxt service list` reports it."""

    name: str
    base_path: PxtPath  # the catalog directory the service's models are bound to
    endpoint: str
    pid: int
    created_at: float
    app_file: str  # the file the service was served from


def delete_service_op(name: str, endpoint: Optional[str], status: OpStatus) -> ServiceChangeOp:
    """The operation for deleting the named deployment, in the given status."""
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
    'ServiceChangeOp',
    'ServiceDeployment',
    'ServiceDiff',
    'ServicePlan',
    'ServicePlanSummary',
    'ServiceResolution',
    'Severity',
    'delete_service_op',
]

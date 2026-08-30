"""Shared types between the CLI and pixeltable core."""

from __future__ import annotations

import uuid
from typing import Any, Literal

import pydantic

from pixeltable_cli.utils import PxtPath

OpStatus = Literal['applied', 'skipped', 'refused', 'failed']

Severity = Literal['additive', 'destructive', 'unsupported', 'blocked']

Resolution = Literal['up_to_date', 'create', 'update_additive', 'update_destructive', 'unsupported', 'blocked']


def _summary(changes: list[str]) -> str:
    """The first few causes, printable, with the rest counted."""
    if len(changes) <= 3:
        return '; '.join(changes)
    return f'{"; ".join(changes[:3])} and {len(changes) - 3} more'


class ChangeOp(pydantic.BaseModel):
    """One reconciliation operation against a target."""

    # what the operation acts on, as communicated to the user (a column, a route, a secret key, a field, etc.)
    name: str | None

    op: Literal['add', 'drop', 'alter']
    severity: Severity
    description: str  # one sentence, ready to print

    requires_restart: bool = False  # whether applying this interrupts what is running
    status: OpStatus | None = None  # the outcome of the operation

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def destructive(self) -> bool:
        return self.severity == 'destructive'


class SchemaChangeIndexRef(pydantic.BaseModel):
    index_type: Literal['btree', 'embedding']
    columns: list[str]
    name: str | None


class SchemaChangeOpDetails(pydantic.BaseModel):
    """Operands of a SchemaChangeOp, rendered as strings to survive serialization"""

    type: str | None = None  # the new type for a column add or alter
    value: str | None = None  # the new computed value expression for a column add or alter
    index_ref: SchemaChangeIndexRef | None = None  # the new index for an index add or alter


class SchemaChangeOp(ChangeOp):
    """
    A single schema change operation (eg, add column, drop column, etc).
    """

    target: Literal['column', 'index', 'table']

    details: SchemaChangeOpDetails = pydantic.Field(default_factory=SchemaChangeOpDetails)

    # excluded from serialization: an expr or a column type does not survive it
    model: Any = pydantic.Field(default=None, exclude=True)  # model-side value; None for drops
    existing: Any = pydantic.Field(default=None, exclude=True)  # catalog-side value; None for adds

    @classmethod
    def drop_table(cls, pxt_path: PxtPath, status: OpStatus) -> SchemaChangeOp:
        """The operation for dropping the table at the given path, in the given status."""
        return cls(
            target='table',
            name=pxt_path,
            op='drop',
            severity='destructive',
            description=f'table {pxt_path!r} will be dropped',
            status=status,
        )


class ServiceChangeOp(ChangeOp):
    """One operation reconciling a running service with a ServiceSpec."""

    target: Literal['service', 'base_path', 'route', 'resources', 'secret', 'project']

    details: dict[str, str] = pydantic.Field(default_factory=dict)

    @classmethod
    def otel(cls, current: bool, requested: bool) -> ServiceChangeOp:
        state = {True: 'on', False: 'off'}
        return cls(
            target='service',
            name='otel',
            op='alter',
            severity='additive',  # the routes are unchanged; the service restarts to pick up the new setting
            description=f'tracing will be turned {state[requested]}, which restarts the service',
            details={'from': state[current], 'to': state[requested]},
            requires_restart=True,
        )

    @classmethod
    def project_moved(cls, changes: list[str], command: str | None = None) -> ServiceChangeOp:
        """The operation for a project that moved on since the instance started.

        changes are the causes, from ProjectFingerprint.changes(). With a command, the instance cannot be
        brought up to date by restarting it -- a hosted service runs the project its database was given, not
        the project here -- so the operation is blocked.
        """
        summary = _summary(changes)
        if command is None:
            return cls(
                target='project',
                name='project',
                op='alter',
                severity='additive',  # what the service serves is unchanged; it restarts to run the new code
                description=f'{summary}, which restarts the service',
                details={'changes': '; '.join(changes)},
                requires_restart=True,
            )
        return cls(
            target='project',
            name='project',
            op='alter',
            severity='blocked',
            description=f'{summary}; run {command} to upload them',
            details={'changes': '; '.join(changes), 'command': command},
        )

    @classmethod
    def db_not_updated(cls, command: str) -> ServiceChangeOp:
        """The operation for a database `pxt db update` has not run for."""
        return cls(
            target='project',
            name='project',
            op='alter',
            severity='blocked',
            description=f'the database has nothing to serve; run {command}',
            details={'command': command},
        )

    @classmethod
    def blocked_schema(cls, service_name: str, description: str, command: str) -> ServiceChangeOp:
        return cls(
            target='service',
            name=service_name,
            op='alter',
            severity='blocked',
            description=description,
            details={'command': command},
        )

    @classmethod
    def prefix_moved(cls, service_name: str, current_prefix: str, declared_prefix: str) -> ServiceChangeOp:
        return cls(
            target='service',
            name=service_name,
            op='alter',
            severity='destructive',
            description=(
                f'service {service_name!r} will be served at prefix {declared_prefix!r} rather than {current_prefix!r}'
            ),
            details={'from': current_prefix, 'to': declared_prefix},
            requires_restart=True,
        )

    @classmethod
    def route_added(cls, name: str) -> ServiceChangeOp:
        return cls(
            target='route',
            name=name,
            op='add',
            severity='additive',
            description=f'route {name!r} will be added',
            requires_restart=True,
        )

    @classmethod
    def route_replaced(cls, name: str, changed: list[str]) -> ServiceChangeOp:
        return cls(
            target='route',
            name=name,
            op='alter',
            severity='destructive',
            description=f'route {name!r} will be replaced: {", ".join(changed)} changed',
            details={'changed': ', '.join(changed)},
            requires_restart=True,
        )

    @classmethod
    def route_dropped(cls, name: str) -> ServiceChangeOp:
        return cls(
            target='route',
            name=name,
            op='drop',
            severity='destructive',
            description=f'route {name!r} will no longer be served',
            requires_restart=True,
        )

    @classmethod
    def path_added(cls, path: str) -> ServiceChangeOp:
        return cls(
            target='route',
            name=path,
            op='add',
            severity='additive',
            description=f'path {path!r} will be served',
            requires_restart=True,
        )

    @classmethod
    def path_dropped(cls, path: str) -> ServiceChangeOp:
        return cls(
            target='route',
            name=path,
            op='drop',
            severity='destructive',
            description=f'path {path!r} will no longer be served',
            requires_restart=True,
        )

    @classmethod
    def delete_service(cls, name: str, endpoint: str | None, status: OpStatus) -> ServiceChangeOp:
        """The operation for deleting the named service, in the given status."""
        served = '' if endpoint is None else f' at {endpoint}'
        return cls(
            target='service',
            name=name,
            op='drop',
            severity='destructive',
            description=f'service {name!r}{served} will be deleted',
            details={} if endpoint is None else {'endpoint': endpoint},
            status=status,
        )


# what a DbChangeOp acts on. The two artifacts are separate: 'image' is the environment the pods run on,
# 'archive' the sources they fetch, and a source edit moves only the second.
DbTarget = Literal['image', 'archive', 'capacity', 'secret', 'placement']


class DbChangeOp(ChangeOp):
    """One operation reconciling a hosted database with a DatabaseConfig."""

    target: DbTarget

    details: dict[str, str] = pydantic.Field(default_factory=dict)

    @classmethod
    def image_moved(cls, changes: list[str]) -> DbChangeOp:
        """The operation for an environment that differs from the one the current image holds.

        changes are the causes, from ProjectFingerprint.changes().
        """
        return cls(
            target='image',
            name='image',
            op='alter',
            # the image is replaced, not removed: what the database serves is unchanged until a pod restarts
            severity='additive',
            description=f'the image will be rebuilt: {_summary(changes)}',
            details={'changes': '; '.join(changes)},
            requires_restart=True,
        )

    @classmethod
    def archive_moved(cls, changes: list[str]) -> DbChangeOp:
        """The operation for sources the database's pods are not running.

        changes are the causes, from ProjectFingerprint.changes().
        """
        return cls(
            target='archive',
            name='project',
            op='alter',
            # what the pods serve is unchanged; they restart to run the new sources
            severity='additive',
            description=f'the project files will be uploaded: {_summary(changes)}',
            details={'changes': '; '.join(changes)},
            requires_restart=True,
        )

    @classmethod
    def capacity(cls, field: str, current: float | int | None, declared: float | int) -> DbChangeOp:
        was = 'unreported' if current is None else str(current)
        return cls(
            target='capacity',
            name=field,
            op='alter',
            severity='destructive' if current is not None and declared < current else 'additive',
            description=f'{field} will be {declared} rather than {was}, which restarts the database',
            details={'from': was, 'to': str(declared)},
            requires_restart=True,
        )

    @classmethod
    def placement(cls, field: str, current: str, declared: str) -> DbChangeOp:
        """The operation for a field fixed at creation, which no update can carry out."""
        return cls(
            target='placement',
            name=field,
            op='alter',
            severity='unsupported',
            description=(
                f'{field} is {current!r} and cannot be changed to {declared!r}; create a database there instead'
            ),
            details={'from': current, 'to': declared},
        )

    @classmethod
    def secret(cls, key: str, op: Literal['add', 'drop']) -> DbChangeOp:
        if op == 'add':
            return cls(
                target='secret', name=key, op='add', severity='additive', description=f'secret {key!r} will be set'
            )
        return cls(
            target='secret',
            name=key,
            op='drop',
            severity='destructive',
            description=f'secret {key!r} will be deleted, and code reading it will fail',
        )

    @classmethod
    def build_image(cls) -> DbChangeOp:
        """The operation for an image build the caller asked for rather than one a difference calls for."""
        return cls(
            target='image',
            name='image',
            op='alter',
            severity='additive',
            description="the image will be rebuilt from the project's environment",
            requires_restart=True,
        )

    @classmethod
    def upload_archive(cls) -> DbChangeOp:
        """The operation for uploading the project the caller named rather than one a difference calls for."""
        return cls(
            target='archive',
            name='project',
            op='alter',
            severity='additive',
            description='the project files will be uploaded',
            requires_restart=True,
        )


# Schema


class TableDiff(pydantic.BaseModel):
    """How one model differs from its catalog table."""

    path: str  # catalog path of the table
    model_cls: str  # model class name, so an agent can map back to code
    kind: Literal['table', 'view']
    exists: bool
    resolution: Resolution

    # empty for a create, which subsumes the additions that constitute it
    ops: list[SchemaChangeOp] = pydantic.Field(default_factory=list)

    # identity of the existing table, as of the read this diff was computed from; None if it doesn't exist yet
    tbl_id: uuid.UUID | None = pydantic.Field(default=None, exclude=True)

    # schema versions of the TableVersionPath
    schema_versions: dict[uuid.UUID, int] | None = pydantic.Field(default=None, exclude=True)

    status: OpStatus | None = None

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def destructive(self) -> bool:
        return any(op.destructive for op in self.ops)


class SchemaPlanSummary(pydantic.BaseModel):
    up_to_date: int
    create: int
    update_additive: int
    update_destructive: int
    unsupported: int
    extras: int
    destructive: int  # operations, not tables


class SchemaPlan(pydantic.BaseModel):
    """Set of changes needed to reconcile a target directory with a schema model."""

    app_file: str
    catalog_dir: PxtPath
    tables: list[TableDiff] = pydantic.Field(default_factory=list)
    extras: list[PxtPath] = pydantic.Field(default_factory=list)  # tables under catalog_dir no model declares

    ops: list[SchemaChangeOp] = pydantic.Field(default_factory=list)  # on whole tables, unlike TableDiff.ops
    status: OpStatus | None = None

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def in_agreement(self) -> bool:
        """True if no table needs a create or an update; extras don't count."""
        return all(t.resolution == 'up_to_date' for t in self.tables)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> SchemaPlanSummary:
        return SchemaPlanSummary(
            up_to_date=self._count('up_to_date'),
            create=self._count('create'),
            update_additive=self._count('update_additive'),
            update_destructive=self._count('update_destructive'),
            unsupported=self._count('unsupported'),
            extras=len(self.extras),
            destructive=sum(1 for t in self.tables for op in t.ops if op.destructive),
        )

    def _count(self, resolution: Resolution) -> int:
        return sum(1 for t in self.tables if t.resolution == resolution)


# Services


class RouteSpec(pydantic.BaseModel):
    """Complete metadata of a service route.

    Location-independent: a route declared against a model names the model, never the catalog path the model
    is bound to. Two routes with equal specifications serve the same contract.

    This is also used for display purposes and must never contain any cleartext secrets.
    """

    method: Literal['GET', 'POST']

    # as served, including every prefix in front of the declared path
    path: str
    route_type: Literal['insert', 'update', 'delete', 'compute', 'query']

    # at most one is set: model holds the table name a model declares, table the catalog path of a table
    model: str | None = None
    table: str | None = None

    # for a query route, inputs are its query's parameters and outputs the fields of its response
    inputs: list[str] = pydantic.Field(default_factory=list)
    uploadfile_inputs: list[str] = pydantic.Field(default_factory=list)
    outputs: list[str] = pydantic.Field(default_factory=list)
    match_columns: list[str] = pydantic.Field(default_factory=list)

    background: bool = False
    return_fileresponse: bool = False
    one_row: bool = False
    export_sql: dict[str, Any] | None = None  # the export target of a route, from SqlExport.display_dict()
    query: str | None = None  # the symbol path of the query udf


class ServiceSpec(pydantic.BaseModel):
    """Complete metadata of a service."""

    name: str
    routes: list[RouteSpec] = pydantic.Field(default_factory=list)

    # paths served by a custom FastAPI instance that don't come from FastAPIRouter instances
    app_paths: list[str] = pydantic.Field(default_factory=list)


# How the routes were compared. 'declarative' compares the route declarations the running service was started
# from; 'openapi' compares the OpenAPI document generated from a custom application. 'unavailable' means the
# comparison did not happen, which is not the same as happening and finding no differences.
RouteComparison = Literal['declarative', 'openapi', 'unavailable']


class ServiceDiff(pydantic.BaseModel):
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

    resolution: Resolution

    route_comparison: RouteComparison
    route_detail: str | None  # why the routes were not compared, when they were not

    # empty for a create, which subsumes the additions that constitute it
    ops: list[ServiceChangeOp] = pydantic.Field(default_factory=list)

    status: OpStatus | None = None

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def destructive(self) -> bool:
        return any(op.destructive for op in self.ops)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def requires_restart(self) -> bool:
        return self.exists and any(op.requires_restart for op in self.ops)


class ServicePlanSummary(pydantic.BaseModel):
    up_to_date: int
    create: int
    update_additive: int
    update_destructive: int
    extras: int
    blocked: int  # services whose reconciliation the database has to enable first
    destructive: int  # operations, not services
    blocked_ops: int  # operations the database has to satisfy before the plan can be applied
    restarts: int  # services that applying the plan would interrupt


class ServicePlan(pydantic.BaseModel):
    """Set of changes needed to reconcile the services at a target with the definitions a file holds."""

    app_file: str

    # the database the services live in, and the catalog path their models bind against
    target: PxtPath

    services: list[ServiceDiff] = pydantic.Field(default_factory=list)
    extras: list[str] = pydantic.Field(default_factory=list)  # services at the target the file does not declare

    ops: list[ServiceChangeOp] = pydantic.Field(default_factory=list)  # on whole services, unlike ServiceDiff.ops
    status: OpStatus | None = None

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def in_agreement(self) -> bool:
        """True if no service needs a create or an update; extras don't count."""
        return all(d.resolution == 'up_to_date' for d in self.services)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> ServicePlanSummary:
        return ServicePlanSummary(
            up_to_date=self._count('up_to_date'),
            create=self._count('create'),
            update_additive=self._count('update_additive'),
            update_destructive=self._count('update_destructive'),
            extras=len(self.extras),
            blocked=self._count('blocked'),
            destructive=sum(1 for d in self.services for op in d.ops if op.destructive),
            blocked_ops=sum(1 for d in self.services for op in d.ops if op.severity == 'blocked'),
            restarts=sum(1 for d in self.services if d.requires_restart),
        )

    def _count(self, resolution: Resolution) -> int:
        return sum(1 for d in self.services if d.resolution == resolution)


class ServiceInstance(pydantic.BaseModel):
    """A service instance, as `pxt service list` shows it."""

    name: str
    base_path: PxtPath  # the catalog directory the service's models are bound to
    endpoint: str
    state: str
    error: str | None  # why a FAILED instance failed, when its manager reports a reason
    app_module: str  # dotted module path, relative to the project root
    spec: ServiceSpec  # what it serves

    pid: int | None  # the process serving the instance; set only for an instance running on this machine
    process_started_at: float | None  # creation time of pid, None where the platform does not report one


# Databases


class DbPlanSummary(pydantic.BaseModel):
    ops: int
    destructive: int
    unsupported: int
    rebuild: bool  # whether the plan rebuilds the image, which is the one step that takes minutes
    restarts: bool  # whether applying the plan interrupts what the database is serving


class DbPlan(pydantic.BaseModel):
    """Set of changes needed to reconcile a hosted database with the entry in a project that declares it."""

    db_uri: str
    exists: bool
    state: str | None  # the database's state, None when it does not exist
    resolution: Resolution
    ops: list[DbChangeOp] = pydantic.Field(default_factory=list)
    status: OpStatus | None = None

    @classmethod
    def from_ops(cls, db_uri: str, state: str | None, ops: list[DbChangeOp]) -> DbPlan:
        """The plan the given operations describe; a state of None is a database that does not exist."""
        resolution: Resolution
        if state is None:
            resolution = 'create'
        elif any(op.severity == 'unsupported' for op in ops):
            resolution = 'unsupported'
        elif any(op.destructive for op in ops):
            resolution = 'update_destructive'
        elif len(ops) > 0:
            resolution = 'update_additive'
        else:
            resolution = 'up_to_date'
        return cls(db_uri=db_uri, exists=state is not None, state=state, resolution=resolution, ops=ops)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def in_agreement(self) -> bool:
        return self.resolution == 'up_to_date'

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def destructive(self) -> bool:
        return any(op.destructive for op in self.ops)

    @pydantic.computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> DbPlanSummary:
        return DbPlanSummary(
            ops=len(self.ops),
            destructive=sum(1 for op in self.ops if op.destructive),
            unsupported=sum(1 for op in self.ops if op.severity == 'unsupported'),
            rebuild=any(op.target == 'image' for op in self.ops),
            restarts=any(op.requires_restart for op in self.ops),
        )


class CheckReport(pydantic.BaseModel):
    """What checking a file reports: whether it is valid, and what to fix or to know about."""

    file: str
    valid: bool  # False when errors is non-empty
    errors: list[str] = pydantic.Field(default_factory=list)
    warnings: list[str] = pydantic.Field(default_factory=list)

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field

from pixeltable_cli.utils import OpStatus, validate_path_shape


def _validate_pxt_path(v: str | None) -> str | None:
    if v is None or v == '':
        return v
    err = validate_path_shape(v)
    if err is not None:
        raise ValueError(err)
    return v


PxtPath = Annotated[str, AfterValidator(_validate_pxt_path)]


class HealthResponse(BaseModel):
    ok: bool
    service: Literal['pxt'] = 'pxt'
    pxt_version: str
    pid: int
    started_at: str

    # Identity fingerprint: every field below is captured once at daemon startup and reported
    # verbatim on each /health call. The client computes the same fingerprint locally (without
    # importing pixeltable) and restarts the daemon on any mismatch, so the daemon never keeps
    # serving requests against a stale install or a stale snapshot of the environment.
    pxt_install_dir: str
    python_executable: str
    pixeltable_home: str
    pixeltable_pgdata: str
    pixeltable_config_file: str

    # PIXELTABLE_*-prefixed env vars at daemon-startup time. Values for keys naming a
    # credential are replaced with a sha256 prefix so /health doesn't leak secrets; the
    # client redacts the same way so equal plaintexts still produce equal entries.
    pixeltable_env: dict[str, str]


class LsEntry(BaseModel):
    path: str
    kind: Literal['table', 'view', 'snapshot', 'replica', 'dir']
    num_rows: int | None = None
    num_cols: int | None = None
    last_version: int | None = None
    flags: str = ''


class LsResponse(BaseModel):
    entries: list[LsEntry]
    tree: dict[str, Any] | None = None


class DescribeResponse(BaseModel):
    text: str
    metadata: dict[str, Any]


class ErrorEntry(BaseModel):
    pk: dict[str, Any]
    column: str
    errortype: str
    errormsg: str | None


class ErrorsResponse(BaseModel):
    entries: list[ErrorEntry]


class HistoryResponse(BaseModel):
    versions: list[dict[str, Any]]  # raw VersionMetadata; client formats


class ColumnEntry(BaseModel):
    table: str
    column: str
    is_computed: bool
    type_: str
    computed_with: str | None = None
    depends_on: list[tuple[str, str]] | None = None


class ColumnsResponse(BaseModel):
    entries: list[ColumnEntry]


class IdxEntry(BaseModel):
    table: str
    name: str
    columns: list[str]
    index_type: str
    metric: str | None = None
    embedding: str | None = None


class IdxsResponse(BaseModel):
    entries: list[IdxEntry]


class RowsResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]


class StatusResponse(BaseModel):
    pxt_version: str
    pid: int
    started_at: str
    home: str | None
    db_url: str | None = Field(default=None, description='Database URL with credentials redacted.')
    media_dir: str | None
    file_cache_dir: str | None
    media_size_bytes: int | None = Field(
        default=None, description='Populated only when the client requests sizes=1; otherwise None.'
    )
    file_cache_size_bytes: int | None = None
    total_tables: int
    total_errors: int


class ConfigEntry(BaseModel):
    section: str
    key: str
    value: str | None = Field(
        description="Resolved value as a string, or None if unset. '<redacted>' for sensitive keys."
    )
    source: str = Field(
        description="'env' if the value came from an environment variable or programmatic override, "
        "'unset' if no source carries it, or the absolute path of the file the value was loaded from."
    )
    description: str
    expected_type: str


class ConfigResponse(BaseModel):
    config_file: str
    entries: list[ConfigEntry]


class CountResponse(BaseModel):
    path: str
    count: int


class GetResponse(BaseModel):
    pk_columns: list[str]
    row: dict[str, Any] | None


class DropBody(BaseModel):
    cascade: bool = False  # drop dependent views (tables) or recurse (dirs)


class DropResponse(BaseModel):
    path: str
    dropped: bool


class MoveBody(BaseModel):
    path: PxtPath
    new_path: PxtPath


class MoveResponse(BaseModel):
    path: str
    new_path: str


class RevertBody(BaseModel):
    steps: int = 1  # number of consecutive revert() calls


class RevertResponse(BaseModel):
    path: str
    from_version: int
    to_version: int


class SchemaDiffBody(BaseModel):
    schema_path: str  # absolute filesystem path to the schema file on the daemon host
    target: PxtPath


# The schema-plan models below type the responses that carry a schema plan; utils.SchemaPlan documents the format
# itself. Two mappings out of catalog.model are hand-written and have to be changed together with it:
#   - `action` here <-> bridge._ACTIONS, keyed by catalog.model.DiffResolution
#   - an op's `kind` <-> bridge._OP_KINDS, keyed by (SchemaChangeOp.target, SchemaChangeOp.op)
# Both fail loudly on a mismatch: a KeyError in the bridge, or pydantic rejecting the response. An op's `severity`
# passes through from SchemaChangeOp unvalidated, and commands/schema.py:_severity_label() renders it.


class SchemaPlanOp(BaseModel):
    kind: str  # 'add_column', 'drop_index', 'drop_table', ...
    name: str  # what the operation acts on: a column, an index, a table attribute, or a table path
    severity: str
    destructive: bool
    description: str
    details: dict[str, str]  # the operands of this kind of operation, eg 'type' for add_column


class SchemaDiffTable(BaseModel):
    path: str  # catalog path of the table
    model_cls: str  # model class name, so an agent can map back to code
    kind: Literal['table', 'view']
    action: Literal['create', 'update', 'noop', 'unsupported']
    destructive: bool
    # empty for a create, which subsumes the additions that constitute it. Declared covariantly, so that a
    # response whose operations all carry a status can narrow it.
    ops: Sequence[SchemaPlanOp]


class SchemaDiffSummary(BaseModel):
    create: int
    update: int
    noop: int
    unsupported: int
    extras: int
    destructive: int  # number of destructive operations, across all tables


class SchemaDiffResponse(BaseModel):
    # 'schema_path', not 'schema': a field named 'schema' shadows an attribute of pydantic's BaseModel
    schema_path: str
    target: str
    in_agreement: bool  # True if no table needs a create or an update; extras don't count
    tables: list[SchemaDiffTable]
    extras: list[str]  # tables under the target that no model declares
    summary: SchemaDiffSummary


class SchemaPruneBody(BaseModel):
    schema_path: str  # absolute filesystem path to the schema file on the daemon host
    target: PxtPath


class SchemaAppliedOp(SchemaPlanOp):
    status: OpStatus


class SchemaPruneResponse(SchemaDiffResponse):
    ops: list[SchemaAppliedOp]  # one drop_table operation per table dropped


class SchemaUpdateBody(BaseModel):
    schema_path: str  # absolute filesystem path to the schema file on the daemon host
    target: PxtPath
    allow_destructive: bool = False


class SchemaUpdateTable(SchemaDiffTable):
    status: OpStatus
    ops: Sequence[SchemaAppliedOp]


class SchemaUpdateResponse(BaseModel):
    # the plan that was applied: the same shape 'schema diff' returns, with a status on each table and operation
    schema_path: str
    target: str
    in_agreement: bool
    tables: list[SchemaUpdateTable]
    extras: list[str]
    summary: SchemaDiffSummary

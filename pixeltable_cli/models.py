from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field

from pixeltable_cli import utils


def _validate_pxt_path(v: str | None) -> str | None:
    if v is None or v == '':
        return v
    err = utils.validate_path_shape(v)
    if err is not None:
        raise ValueError(err)
    return v


PxtPath = Annotated[utils.PxtPath, AfterValidator(_validate_pxt_path)]


def _validate_db_uri(v: str) -> str:
    parts = utils.split_pxt_uri(v)
    if parts is None or parts.db is None or parts.path is not None:
        raise ValueError(f'{v!r} does not name a hosted database; write pxt://org:db')
    return v


# the uri of a hosted database, checked here so that every db verb refuses a bad one the same way
DbUri = Annotated[str, AfterValidator(_validate_db_uri)]


# the verbs the daemon dispatches
Method = Literal['GET', 'POST']


class InFlightRequest(BaseModel):
    method: Method
    path: str
    started_at: float


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
    project_root: str | None = None

    # requests being served right now, oldest first
    in_flight: list[InFlightRequest] = Field(default_factory=list)

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
    project_root: str | None = Field(default=None, description='Project root of the daemon, if any.')
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

    # {env var name: hash of the value}
    env_fingerprint: dict[str, str] = Field(default_factory=dict)
    # all recognized env vars
    env_var_names: list[str] = Field(default_factory=list)


class CountResponse(BaseModel):
    path: str
    count: int


class GetResponse(BaseModel):
    pk_columns: list[str]
    row: dict[str, Any] | None


class DropBody(BaseModel):
    path: PxtPath
    cascade: bool = False  # drop dependent views (tables) or recurse (dirs)


class DropResponse(BaseModel):
    path: str
    dropped: bool


class MoveBody(BaseModel):
    path: PxtPath
    new_path: PxtPath
    dry_run: bool = False  # resolve both paths and report them, without moving anything


class MoveResponse(BaseModel):
    path: str
    new_path: str


class RevertBody(BaseModel):
    path: PxtPath
    steps: int = 1  # number of consecutive revert() calls


class RevertResponse(BaseModel):
    path: str
    from_version: int
    to_version: int


class SchemaCheckBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host


class ServiceCheckBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host


class SchemaDiffBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    catalog_dir: PxtPath


class SchemaPruneBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    catalog_dir: PxtPath


class SchemaUpdateBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    catalog_dir: PxtPath
    allow_destructive: bool = False


class ServiceDiffBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    target: PxtPath  # the catalog directory the services' models bind against
    otel: bool = False  # compares the instances against this tracing setting


class ServicePruneBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    target: PxtPath
    dry_run: bool = False


class ServiceUpdateBody(BaseModel):
    app_file: str  # absolute filesystem path to the application file on the daemon host
    target: PxtPath
    allow_destructive: bool = False
    otel: bool = False


class DbDiffBody(BaseModel):
    db_uri: DbUri


class DbUpdateBody(BaseModel):
    db_uri: DbUri
    allow_destructive: bool = False


class DbBuildImageBody(BaseModel):
    db_uri: DbUri


class ServiceStopBody(BaseModel):
    # each one an address ('pxt://org:db/dir/ingest', 'dir/ingest') or a bare local service name
    names: list[str]


class CwdBody(BaseModel):
    uri: str


class CwdResponse(BaseModel):
    uri: str | None  # the session's working directory, or None when unset (catalog root)

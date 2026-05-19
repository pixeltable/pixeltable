from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, Field


def _slash_only(v: str | None) -> str | None:
    # pxt paths are slash-separated and relative. '.' is reserved (pixeltable's legacy
    # separator); leading/trailing '/' or '//' would yield empty components that pixeltable
    # rejects later with a generic "Invalid path" error - reject them here for a clear message.
    if v is None or v == '':
        return v
    if '.' in v:
        raise ValueError(f"pxt paths use '/' as the separator; got {v!r}")
    if v.startswith('/'):
        raise ValueError(f"pxt paths are relative; drop the leading '/' (use '' for root). Got {v!r}")
    if v.endswith('/'):
        raise ValueError(f"pxt paths must not end with '/'; got {v!r}")
    if '//' in v:
        raise ValueError(f"pxt paths must not contain empty components ('//'); got {v!r}")
    return v


PxtPath = Annotated[str, AfterValidator(_slash_only)]


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
    source: Literal['env', 'file', 'unset']
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


class DashboardControlBody(BaseModel):
    action: Literal['enable', 'disable']


class DashboardControlResponse(BaseModel):
    enabled: bool

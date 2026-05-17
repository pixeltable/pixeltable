from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, Field


def _slash_only(v: str | None) -> str | None:
    # pcli paths are slash-separated and relative. '.' is reserved (pixeltable's legacy
    # separator); leading/trailing '/' or '//' would yield empty components that pixeltable
    # rejects later with a generic "Invalid path" error - reject them here for a clear message.
    if v is None or v == '':
        return v
    if '.' in v:
        raise ValueError(f"pcli paths use '/' as the separator; got {v!r}")
    if v.startswith('/'):
        raise ValueError(f"pcli paths are relative; drop the leading '/' (use '' for root). Got {v!r}")
    if v.endswith('/'):
        raise ValueError(f"pcli paths must not end with '/'; got {v!r}")
    if '//' in v:
        raise ValueError(f"pcli paths must not contain empty components ('//'); got {v!r}")
    return v


PcliPath = Annotated[str, AfterValidator(_slash_only)]
OptionalPcliPath = Annotated[str | None, AfterValidator(_slash_only)]


class HealthResponse(BaseModel):
    ok: bool
    service: Literal['pcli'] = 'pcli'
    pxt_version: str
    pid: int
    started_at: str


class LsEntry(BaseModel):
    path: str
    kind: Literal['table', 'view', 'snapshot', 'replica', 'dir']
    num_rows: int | None = None
    num_cols: int | None = None
    last_version: int | None = None
    flags: str = ''


class LsRequest(BaseModel):
    path: PcliPath = ''
    tree: bool = False
    details: bool = False  # populate num_cols/flags (one get_metadata() per entry)
    counts: bool = False  # opt-in: row counts run queries, so off by default


class LsResponse(BaseModel):
    entries: list[LsEntry]
    tree: dict | None = None


class DescribeRequest(BaseModel):
    path: PcliPath


class DescribeResponse(BaseModel):
    text: str
    metadata: dict


class ErrorsRequest(BaseModel):
    path: PcliPath
    col: str | None = None


class ErrorEntry(BaseModel):
    pk: dict
    column: str
    errortype: str
    errormsg: str | None


class ErrorsResponse(BaseModel):
    entries: list[ErrorEntry]


class HistoryRequest(BaseModel):
    path: PcliPath
    n: int | None = None


class HistoryResponse(BaseModel):
    versions: list[dict]  # raw VersionMetadata; client formats


class ColumnsRequest(BaseModel):
    path: OptionalPcliPath = None
    computed_only: bool = False


class ColumnEntry(BaseModel):
    table: str
    column: str
    is_computed: bool
    type_: str
    computed_with: str | None = None
    depends_on: list[tuple[str, str]] | None = None


class ColumnsResponse(BaseModel):
    entries: list[ColumnEntry]


class IdxsRequest(BaseModel):
    path: OptionalPcliPath = None
    embedding_only: bool = False


class IdxEntry(BaseModel):
    table: str
    name: str
    columns: list[str]
    index_type: str
    metric: str | None = None
    embedding: str | None = None


class IdxsResponse(BaseModel):
    entries: list[IdxEntry]


class RowsRequest(BaseModel):
    path: PcliPath
    n: int = 10
    cols: list[str] | None = None


class RowsResponse(BaseModel):
    columns: list[str]
    rows: list[dict]


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


class EnvResponse(BaseModel):
    env_vars: dict[str, str]
    config_file: str | None
    credentials_present: dict[str, bool] = Field(
        description='Presence-only map for agent-relevant credential vars (no values).'
    )


class CountRequest(BaseModel):
    path: PcliPath


class CountResponse(BaseModel):
    path: str
    count: int


class GetRequest(BaseModel):
    path: PcliPath
    pk: list  # values in PK column order
    cols: list[str] | None = None


class GetResponse(BaseModel):
    pk_columns: list[str]
    row: dict | None


class DropRequest(BaseModel):
    path: PcliPath
    cascade: bool = False  # drops dependent views (tables) or recurses (dirs)
    is_dir: bool  # client tells us which API to call


class DropResponse(BaseModel):
    path: str
    dropped: bool


class MoveRequest(BaseModel):
    path: PcliPath
    new_path: PcliPath


class MoveResponse(BaseModel):
    path: str
    new_path: str


class RevertRequest(BaseModel):
    path: PcliPath
    steps: int = 1  # number of consecutive revert() calls


class RevertResponse(BaseModel):
    path: str
    from_version: int
    to_version: int

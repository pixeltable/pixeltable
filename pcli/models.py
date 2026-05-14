from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    pxt_version: str
    pid: int
    started_at: str


class LsEntry(BaseModel):
    path: str
    kind: Literal['table', 'view', 'dir']
    num_rows: int | None = None
    num_cols: int | None = None
    last_version: int | None = None
    flags: str = ''


class LsRequest(BaseModel):
    path: str = '/'
    tree: bool = False
    counts: bool = False  # opt-in: row counts run queries, so off by default


class LsResponse(BaseModel):
    entries: list[LsEntry]
    tree: dict | None = None


class DescribeRequest(BaseModel):
    path: str


class DescribeResponse(BaseModel):
    text: str
    metadata: dict


class ErrorsRequest(BaseModel):
    path: str
    col: str | None = None


class ErrorEntry(BaseModel):
    pk: dict
    column: str
    errortype: str
    errormsg: str | None


class ErrorsResponse(BaseModel):
    entries: list[ErrorEntry]


class HistoryRequest(BaseModel):
    path: str
    n: int | None = None


class HistoryResponse(BaseModel):
    versions: list[dict]  # raw VersionMetadata; client formats


class ColumnsRequest(BaseModel):
    path: str | None = None
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
    path: str | None = None
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
    path: str
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
    db_url: str | None
    """Database URL with credentials redacted."""
    media_dir: str | None
    file_cache_dir: str | None
    media_size_bytes: int | None = None
    """Populated only when the client requests sizes=1; otherwise None."""
    file_cache_size_bytes: int | None = None
    total_tables: int
    total_errors: int


class EnvResponse(BaseModel):
    env_vars: dict[str, str]
    config_file: str | None
    credentials_present: dict[str, bool]
    """Presence-only map for agent-relevant credential vars (no values)."""


class CountRequest(BaseModel):
    path: str


class CountResponse(BaseModel):
    path: str
    count: int


class GetRequest(BaseModel):
    path: str
    pk: list  # values in PK column order


class GetResponse(BaseModel):
    pk_columns: list[str]
    row: dict | None


class DropRequest(BaseModel):
    path: str
    cascade: bool = False  # drops dependent views (tables) or recurses (dirs)
    is_dir: bool  # client tells us which API to call


class DropResponse(BaseModel):
    path: str
    dropped: bool


class MoveRequest(BaseModel):
    path: str
    new_path: str


class MoveResponse(BaseModel):
    path: str
    new_path: str


class RevertRequest(BaseModel):
    path: str
    steps: int = 1  # number of consecutive revert() calls


class RevertResponse(BaseModel):
    path: str
    from_version: int
    to_version: int

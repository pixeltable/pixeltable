import datetime
from typing import Literal, Optional, TypedDict


class ColumnMetadata(TypedDict):
    """Metadata for a column of a Pixeltable table."""

    name: str
    """The name of the column."""
    type_: str
    """The type specifier of the column."""
    version_added: int
    """The table version when this column was added."""
    is_stored: bool
    """`True` if this is a stored column; `False` if it is dynamically computed."""
    is_primary_key: bool
    """`True` if this column is part of the table's primary key."""
    media_validation: Optional[Literal['on_read', 'on_write']]
    """The media validation policy for this column."""
    computed_with: Optional[str]
    """Expression used to compute this column; `None` if this is not a computed column."""


class EmbeddingIndexParams(TypedDict):
    metric: Literal['cosine', 'ip', 'l2']
    """Index metric."""
    embeddings: list[str]
    """List of embeddings defined for this index."""


class IndexMetadata(TypedDict):
    """Metadata for a column of a Pixeltable table."""

    name: str
    """The name of the index."""
    columns: list[str]
    """The table columns that are indexed."""
    index_type: Literal['embedding']
    """The type of index (currently only `'embedding'` is supported, but others will be added in the future)."""
    parameters: EmbeddingIndexParams


class TableMetadata(TypedDict):
    """Metadata for a Pixeltable table."""

    name: str
    """The name of the table (ex: `'my_table'`)."""
    path: str
    """The full path of the table (ex: `'my_dir.my_subdir.my_table'`)."""
    columns: dict[str, ColumnMetadata]
    """Column metadata for all of the visible columns of the table."""
    indices: dict[str, IndexMetadata]
    """Index metadata for all of the indices of the table."""
    is_replica: bool
    """`True` if this table is a replica of another (shared) table."""
    is_view: bool
    """`True` if this table is a view."""
    is_snapshot: bool
    """`True` if this table is a snapshot."""
    version: int
    """The current version of the table."""
    version_created: datetime.datetime
    """The timestamp when this table version was created."""
    schema_version: int
    """The current schema version of the table."""
    comment: Optional[str]
    """User-provided table comment, if one exists."""
    media_validation: Literal['on_read', 'on_write']
    """The media validation policy for this table."""
    base: Optional[str]
    """If this table is a view or snapshot, the full path of its base table; otherwise `None`."""


class VersionMetadata(TypedDict):
    """Metadata for a specific version of a Pixeltable table."""

    """The version number."""
    version: int
    """The timestamp when this version was created."""
    created_at: datetime.datetime
    """The user who created this version, if defined."""
    user: str | None
    """The type of table transformation that this version represents (`'data'` or `'schema'`)."""
    change_type: Literal['data', 'schema']
    """The number of rows inserted in this version."""
    inserts: int
    """The number of rows updated in this version."""
    updates: int
    """The number of rows deleted in this version."""
    deletes: int
    """The number of errors encountered during this version."""
    errors: int
    """The number of computed values calculated in this version."""
    computed: int
    """A description of the schema change that occurred in this version, if any."""
    schema_change: str | None

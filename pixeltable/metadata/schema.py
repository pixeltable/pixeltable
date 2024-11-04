import dataclasses
import typing
import uuid
from typing import Any, Optional, TypeVar, Union, get_type_hints

import sqlalchemy as sql
import sqlalchemy.orm as orm
from sqlalchemy import BigInteger, ForeignKey, Integer, LargeBinary
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm.decl_api import DeclarativeMeta

# Base has to be marked explicitly as a type, in order to be used elsewhere as a type hint. But in addition to being
# a type, it's also a `DeclarativeMeta`. The following pattern enables us to expose both `Base` and `Base.metadata`
# outside of the module in a typesafe way.
Base: type = declarative_base()
assert isinstance(Base, DeclarativeMeta)
base_metadata = Base.metadata

T = TypeVar('T')

def md_from_dict(data_class_type: type[T], data: Any) -> T:
    """Re-instantiate a dataclass instance that contains nested dataclasses from a dict."""
    if dataclasses.is_dataclass(data_class_type):
        fieldtypes = {f: t for f, t in get_type_hints(data_class_type).items()}
        return data_class_type(**{f: md_from_dict(fieldtypes[f], data[f]) for f in data})  # type: ignore[return-value]

    origin = typing.get_origin(data_class_type)
    if origin is not None:
        type_args = typing.get_args(data_class_type)
        if origin is Union and type(None) in type_args:
            # Handling Optional types
            non_none_args = [arg for arg in type_args if arg is not type(None)]
            assert len(non_none_args) == 1
            return md_from_dict(non_none_args[0], data) if data is not None else None
        elif origin is list:
            return [md_from_dict(type_args[0], elem) for elem in data]  # type: ignore[return-value]
        elif origin is dict:
            key_type = type_args[0]
            val_type = type_args[1]
            return {key_type(key): md_from_dict(val_type, val) for key, val in data.items()}  # type: ignore[return-value]
        elif origin is tuple:
            return tuple(md_from_dict(arg_type, elem) for arg_type, elem in zip(type_args, data))  # type: ignore[return-value]
        else:
            assert False
    else:
        return data


# structure of the stored metadata:
# - each schema entity that grows somehow proportionally to the data (# of output_rows, total insert operations,
#   number of schema changes) gets its own table
# - each table has an 'md' column that basically contains the payload
# - exceptions to that are foreign keys without which lookups would be too slow (ex.: TableSchemaVersions.tbl_id)
# - the md column contains a dataclass serialized to json; this has the advantage of making changes to the metadata
#   schema easier (the goal is not to have to rely on some schema migration framework; if that breaks for some user,
#   it would be very difficult to patch up)

@dataclasses.dataclass
class SystemInfoMd:
    schema_version: int


class SystemInfo(Base):
    """A single-row table that contains system-wide metadata."""
    __tablename__ = 'systeminfo'
    dummy = sql.Column(Integer, primary_key=True, default=0, nullable=False)
    md = sql.Column(JSONB, nullable=False)  # SystemInfoMd


@dataclasses.dataclass
class DirMd:
    name: str


class Dir(Base):
    __tablename__ = 'dirs'

    id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    parent_id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=True)
    md = sql.Column(JSONB, nullable=False)


@dataclasses.dataclass
class ColumnMd:
    """
    Records the non-versioned metadata of a column.
    - immutable attributes: type, primary key, etc.
    - when a column was added/dropped, which is needed to GC unreachable storage columns
      (a column that was added after table snapshot n and dropped before table snapshot n+1 can be removed
      from the stored table).
     """
    id: int
    schema_version_add: int
    schema_version_drop: Optional[int]
    col_type: dict

    # if True, is part of the primary key
    is_pk: bool

    # if set, this is a computed column
    value_expr: Optional[dict]

    # if True, the column is present in the stored table
    stored: Optional[bool]


@dataclasses.dataclass
class IndexMd:
    """
    Metadata needed to instantiate an EmbeddingIndex
    """
    id: int
    name: str
    indexed_col_id: int  # column being indexed
    index_val_col_id: int  # column holding the values to be indexed
    index_val_undo_col_id: int  # column holding index values for deleted rows
    schema_version_add: int
    schema_version_drop: Optional[int]
    class_fqn: str
    init_args: dict[str, Any]


@dataclasses.dataclass
class ViewMd:
    is_snapshot: bool

    # (table id, version); for mutable views, all versions are None
    base_versions: list[tuple[str, Optional[int]]]

    # filter predicate applied to the base table; view-only
    predicate: Optional[dict[str, Any]]

    # ComponentIterator subclass; only for component views
    iterator_class_fqn: Optional[str]

    # args to pass to the iterator class constructor; only for component views
    iterator_args: Optional[dict[str, Any]]


@dataclasses.dataclass
class TableMd:
    name: str

    # monotonically increasing w/in Table for both data and schema changes, starting at 0
    current_version: int
    # each version has a corresponding schema version (current_version >= current_schema_version)
    current_schema_version: int

    next_col_id: int  # used to assign Column.id
    next_idx_id: int  # used to assign IndexMd.id

    # - used to assign the rowid column in the storage table
    # - every row is assigned a unique and immutable rowid on insertion
    next_row_id: int

    # Metadata format for external stores:
    # {'class': 'pixeltable.io.label_studio.LabelStudioProject', 'md': {'project_id': 3}}
    external_stores: list[dict[str, Any]]

    column_md: dict[int, ColumnMd]  # col_id -> ColumnMd
    index_md: dict[int, IndexMd]  # index_id -> IndexMd
    view_md: Optional[ViewMd]


class Table(Base):
    """
    Table represents both tables and views.

    Views are in essence a subclass of tables, because they also store materialized columns. The differences are:
    - views have a base, which is either a (live) table or a snapshot
    - views can have a filter predicate
    """
    __tablename__ = 'tables'

    MAX_VERSION = 9223372036854775807  # 2^63 - 1

    id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), primary_key=True, nullable=False)
    dir_id: orm.Mapped[uuid.UUID] = orm.mapped_column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=False)
    md = sql.Column(JSONB, nullable=False)  # TableMd


@dataclasses.dataclass
class TableVersionMd:
    created_at: float  # time.time()
    version: int
    schema_version: int


class TableVersion(Base):
    __tablename__ = 'tableversions'
    tbl_id = sql.Column(UUID(as_uuid=True), ForeignKey('tables.id'), primary_key=True, nullable=False)
    version = sql.Column(BigInteger, primary_key=True, nullable=False)
    md = sql.Column(JSONB, nullable=False)  # TableVersionMd


@dataclasses.dataclass
class SchemaColumn:
    """
    Records the versioned metadata of a column.
    """
    pos: int
    name: str

    # media validation strategy of this particular media column; if not set, TableMd.media_validation applies
    # stores column.MediaValiation.name.lower()
    media_validation: Optional[str]


@dataclasses.dataclass
class TableSchemaVersionMd:
    """
    Records all versioned table metadata.
    """
    schema_version: int
    preceding_schema_version: Optional[int]
    columns: dict[int, SchemaColumn]  # col_id -> SchemaColumn
    num_retained_versions: int
    comment: str

    # default validation strategy for any media column of this table
    # stores column.MediaValiation.name.lower()
    media_validation: str


# versioning: each table schema change results in a new record
class TableSchemaVersion(Base):
    __tablename__ = 'tableschemaversions'

    tbl_id = sql.Column(UUID(as_uuid=True), ForeignKey('tables.id'), primary_key=True, nullable=False)
    schema_version = sql.Column(BigInteger, primary_key=True, nullable=False)
    md = sql.Column(JSONB, nullable=False)  # TableSchemaVersionMd


@dataclasses.dataclass
class FunctionMd:
    name: str
    py_version: str  # platform.python_version
    class_name: str  # name of the Function subclass
    md: dict  # part of the output of Function.to_store()


class Function(Base):
    """
    User-defined functions that are not module functions (ie, aren't available at runtime as a symbol in a known
    module).
    Functions without a name are anonymous functions used in the definition of a computed column.
    Functions that have names are also assigned to a database and directory.
    We store the Python version under which a Function was created (and the callable pickled) in order to warn
    against version mismatches.
    """
    __tablename__ = 'functions'

    id = sql.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)
    dir_id = sql.Column(UUID(as_uuid=True), ForeignKey('dirs.id'), nullable=True)
    md = sql.Column(JSONB, nullable=False)  # FunctionMd
    binary_obj = sql.Column(LargeBinary, nullable=True)

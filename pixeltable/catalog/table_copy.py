"""Copy a local base table, with its version history and stored values, to a path in the same database or in a hosted
one.

copy_table() exports the table into chunks of store rows and media files, and passes them to import_table() of the
destination catalog, which creates the table with all of its rows in one transaction. No computed value or embedding
is computed again.

create_table_like() creates an empty table with the columns, indexes and settings of another table.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as FsPath, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import numpy as np
import pgvector.sqlalchemy as sql_vector
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import JSONB

from pixeltable import exceptions as excs, exprs, func, index, type_system as ts
from pixeltable.config import ConfigVar
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.types import ColumnSpec
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, ObjectStoreBase, StorageTarget

from .column import Column
from .globals import IfExistsParam, IfNotExistsParam
from .insertable_table import InsertableTable
from .model.definition import BtreeIndex, EmbeddingIndex, IndexDefinition
from .types import TableVersionKey, TableVersionMd

if TYPE_CHECKING:
    from .catalog import Catalog
    from .catalog_base import CatalogBase
    from .catalog_proxy import CatalogProxy
    from .path import Path
    from .table import Table
    from .table_version import TableVersion

_logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 1_000
# A chunk is a rows file and the media files that its rows are the first to reference. The destination downloads and
# stores one chunk at a time, so these bounds limit its temp disk.
_MAX_CHUNK_ROW_BYTES = 64 * 2**20
_MAX_CHUNK_MEDIA_BYTES = 512 * 2**20
# The export bounds the number of bucket objects in a chunk, since it cannot know an object's size before it downloads
# the object, after the export's transaction.
_MAX_CHUNK_OBJECTS = 64
_MAX_UPLOAD_THREADS = 16
# the number of missing media files an error lists
_MAX_LISTED_FILES = 5


@dataclasses.dataclass(frozen=True)
class _StoreColumn:
    """A stored column of a table. A dropped column is included: its store column still has older row versions."""

    col_id: int
    is_media: bool
    stores_cellmd: bool
    # the table version that added the column; adding a computed column fills the existing rows with files of it
    version_add: int
    # where the column stores its files; None is the default media store
    destination: str | None

    @property
    def store_name(self) -> str:
        return Column.store_name_from_id(self.col_id)

    @property
    def cellmd_store_name(self) -> str:
        return Column.cellmd_store_name_from_id(self.col_id)


def _store_columns(md: list[TableVersionMd]) -> list[_StoreColumn]:
    """The stored columns of the table whose history is md."""
    # the latest schema version that has a column describes it
    schema_cols: dict[int, schema.SchemaColumn] = {}
    for version_md in sorted(md, key=lambda m: m.schema_version_md.schema_version):
        schema_cols.update(version_md.schema_version_md.columns)
    store_cols: list[_StoreColumn] = []
    for col_id, col_md in md[0].tbl_md.column_md.items():
        if not col_md.stored:
            continue
        is_media = ts.ColumnType.from_dict(schema_cols[col_id].col_type).is_media_type()
        store_cols.append(
            _StoreColumn(
                col_id=col_id,
                is_media=is_media,
                stores_cellmd=col_md.stores_cellmd,
                # ColumnMd has the schema version that added the column, not the table version. The earliest table
                # version with that schema version or a later one is the table version that added it.
                version_add=min(
                    m.version_md.version for m in md if m.version_md.schema_version >= col_md.schema_version_add
                ),
                destination=_destination(col_md, schema_cols[col_id], is_media),
            )
        )
    return store_cols


def _destination(col_md: schema.ColumnMd, schema_col: schema.SchemaColumn, is_media: bool) -> str | None:
    """The Column.destination of a stored column. A copy needs it before it writes its metadata, and Column needs a
    TableVersion, which the copy does not have until then."""
    if isinstance(col_md.destination, dict):
        return ConfigVar._from_dict(col_md.destination).value()
    if col_md.destination is not None:
        return col_md.destination
    # an index value column has no name, and its files go to the default media store
    if is_media and schema_col.name is not None:
        if schema_col.value_expr is not None:
            return Env.get().default_output_media_dest
        return Env.get().default_input_media_dest
    return None


def _remap_tbl_id(md: Any, old_id: uuid.UUID, new_id: uuid.UUID) -> Any:
    """Return md with every reference to old_id (a string value equal to str(old_id)) replaced with new_id."""
    old_str, new_str = str(old_id), str(new_id)
    if isinstance(md, str):
        return new_str if md == old_str else md
    if isinstance(md, list):
        return [_remap_tbl_id(x, old_id, new_id) for x in md]
    if isinstance(md, dict):
        return {k: _remap_tbl_id(v, old_id, new_id) for k, v in md.items()}
    return md


def _nested_values(md: Any) -> Iterator[Any]:
    """Yield md and every value that its dicts and lists contain, at any depth."""
    yield md
    if isinstance(md, dict):
        for value in md.values():
            yield from _nested_values(value)
    elif isinstance(md, list):
        for value in md:
            yield from _nested_values(value)


def md_for_copy(md: list[TableVersionMd], tbl_id: uuid.UUID, name: str) -> list[TableVersionMd]:
    """Return the history of a copy of the table whose history is md, with id tbl_id and name name. Raises an error if
    the table cannot be copied into this database."""
    src_md = md[0].tbl_md
    if src_md.view_md is not None:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot copy {src_md.name!r}: only base tables can be copied'
        )
    if src_md.pending_stmt is not None or src_md.tbl_state != schema.TableState.LIVE:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'Cannot copy {src_md.name!r} while an operation on it is in progress; retry the copy',
        )
    # An embedding index builds its functions when the table is loaded, so check those before writing anything. A
    # dropped index is checked as well: a revert of the copy restores it.
    for idx_md in src_md.index_md.values():
        for arg in idx_md.init_args.values():
            if isinstance(arg, dict) and '_classpath' in arg:
                fn = func.Function.from_dict(arg)
                if isinstance(fn, func.InvalidFunction):
                    raise excs.NotFoundError(
                        excs.ErrorCode.FUNCTION_NOT_FOUND,
                        f'Cannot copy {src_md.name!r}: the index {idx_md.name!r} uses the UDF `{fn.self_path}`, '
                        'which is not defined in the destination database.\n'
                        'You can use `pxt db update` to deploy the UDF to the database.',
                    )
    _check_value_expr_udfs(md)

    src_id = uuid.UUID(src_md.tbl_id)
    copy_md = [TableVersionMd.from_dict(_remap_tbl_id(m.as_dict(), src_id, tbl_id)) for m in md]
    tbl_md = copy_md[0].tbl_md
    tbl_md.name = name
    tbl_md.user = Env.get().user
    for col_md in tbl_md.column_md.values():
        if (
            Env.get().is_proxy_daemon
            and isinstance(col_md.destination, str)
            and ObjectPath.parse_object_storage_addr(col_md.destination, False).storage_target
            == StorageTarget.LOCAL_STORE
        ):
            # A proxy daemon receives the metadata of a table of another machine, whose local directory doesn't exist
            # here. The column falls back to the default destination.
            col_md.destination = None
    return copy_md


def _check_value_expr_udfs(md: list[TableVersionMd]) -> None:
    """Raise an error if this database cannot load a UDF of a computed column in md, dropped ones included: a revert of
    the copy restores them."""
    src_name = md[0].tbl_md.name
    for version_md in md:
        for col in version_md.schema_version_md.columns.values():
            # Only the functions of the expression are loaded. Expr.from_dict() cannot be used: the copy has no
            # metadata yet, and an earlier schema version can reference a column that was dropped since. A function
            # with a path is a UDF. A @pxt.query has no path, and its UDFs are nested in it.
            udfs = [
                func.Function.from_dict(node)
                for node in _nested_values(col.value_expr)
                if isinstance(node, dict) and '_classpath' in node and 'path' in node
            ]
            invalid_udf_paths = [udf.self_path for udf in udfs if isinstance(udf, func.InvalidFunction)]
            if len(invalid_udf_paths) > 0:
                # the value column of an index has no name
                col_desc = (
                    f'the computed column {col.name!r}' if col.name is not None else 'the value column of an index'
                )
                raise excs.NotFoundError(
                    excs.ErrorCode.FUNCTION_NOT_FOUND,
                    f'Cannot copy {src_name!r}: {col_desc} cannot be evaluated in the destination database.\n'
                    f'The UDF `{invalid_udf_paths[0]}` is not defined in the destination database.\n'
                    'You can use `pxt db update` to deploy the UDF to the database.',
                )


def _arrow_type(sa_type: sql.types.TypeEngine) -> pa.DataType:
    """The Arrow type of a store column of type sa_type. A store table has columns with no Pixeltable type (rowid,
    v_min, v_max, cellmd and index values), so this maps the SQL types."""
    if isinstance(sa_type, JSONB):
        # json text: the value of a JSONB column has no fixed Arrow type
        return pa.string()
    if isinstance(sa_type, sql.String):
        return pa.string()
    if isinstance(sa_type, sql.Boolean):
        return pa.bool_()
    if isinstance(sa_type, sql.BigInteger):
        return pa.int64()
    if isinstance(sa_type, sql.Float):
        return pa.float64()
    if isinstance(sa_type, sql.TIMESTAMP):
        return pa.timestamp('us', tz='UTC')
    if isinstance(sa_type, sql.Date):
        return pa.date32()
    if isinstance(sa_type, sql.LargeBinary):
        return pa.binary()
    if isinstance(sa_type, sql.UUID):
        return pa.binary(16)
    if isinstance(sa_type, (sql_vector.Vector, sql_vector.HALFVEC)):
        # a vector element is a float32, and a halfvec element is a float16, which float32 represents exactly
        return pa.list_(pa.float32())
    raise AssertionError(f'Unexpected store column type: {sa_type!r}')


def _to_arrow_value(val: Any, sa_type: sql.types.TypeEngine) -> Any:
    if val is None:
        return None
    if isinstance(sa_type, JSONB):
        return json.dumps(val)
    if isinstance(sa_type, sql.UUID):
        return val.bytes
    if isinstance(sa_type, (sql_vector.Vector, sql_vector.HALFVEC)):
        return np.asarray(val, dtype=np.float32)
    return val


def _from_arrow_value(val: Any, sa_type: sql.types.TypeEngine) -> Any:
    if val is None:
        # A Python None inserts a json 'null' into a JSONB column. The store keeps a SQL NULL there.
        return sql.null() if isinstance(sa_type, JSONB) else None
    if isinstance(sa_type, JSONB):
        return json.loads(val)
    if isinstance(sa_type, sql.UUID):
        return uuid.UUID(bytes=val)
    if isinstance(sa_type, (sql_vector.Vector, sql_vector.HALFVEC)):
        return np.array(val, dtype=np.float32)
    return val


@dataclasses.dataclass(frozen=True)
class CopyChunk:
    """A part of a table copy: store rows in the Parquet file rows_file, and the media files that these rows are the
    first to reference (url -> local path)."""

    rows_file: str
    media: dict[str, str]


@dataclasses.dataclass
class _Chunk:
    """The store rows and media files of a table export that become one CopyChunk."""

    rows_file: FsPath
    num_row_bytes: int = 0
    num_media_bytes: int = 0
    # local media files that this chunk is the first to reference: url -> local path
    new_media: dict[str, str] = dataclasses.field(default_factory=dict)
    # urls of the source table's bucket objects that this chunk is the first to reference; they are downloaded into its
    # CopyChunk, and _MAX_CHUNK_OBJECTS bounds their number
    new_objects: set[str] = dataclasses.field(default_factory=set)


class _TableExport:
    """Writes the store rows of a table into chunks, with the media files that the copy must store itself: local
    files, and objects under the table prefix of the source table."""

    md: list[TableVersionMd]
    chunks: list[_Chunk]
    missing_files: list[str]  # urls of local media files that don't exist

    _tmp_dir: FsPath
    _tbl_id: uuid.UUID
    _sa_cols: list[sql.Column]
    _arrow_schema: pa.Schema
    _media_cols: list[str]
    _cellmd_cols: list[str]
    _exported_media: set[str]  # urls in the new_media of a finished chunk
    _chunk: _Chunk | None
    _writer: pq.ParquetWriter | None
    _batch: list[Mapping[str, Any]]

    def __init__(self, md: list[TableVersionMd], sa_tbl: sql.Table, tmp_dir: FsPath):
        self.md = md
        self.chunks = []
        self.missing_files = []
        self._tmp_dir = tmp_dir
        self._tbl_id = uuid.UUID(md[0].tbl_md.tbl_id)
        self._sa_cols = list(sa_tbl.columns)
        self._arrow_schema = pa.schema([(col.name, _arrow_type(col.type)) for col in self._sa_cols])
        cols = _store_columns(md)
        self._media_cols = [col.store_name for col in cols if col.is_media]
        self._cellmd_cols = [col.cellmd_store_name for col in cols if col.stores_cellmd]
        self._exported_media = set()
        self._chunk = None
        self._writer = None
        self._batch = []

    def add_row(self, row: Mapping[str, Any]) -> None:
        if self._chunk is None:
            self._chunk = _Chunk(rows_file=self._tmp_dir / f'rows_{len(self.chunks)}.parquet')
        for name in self._media_cols:
            if row[name] is not None:
                self._add_media_ref(row[name])
        for name in self._cellmd_cols:
            cellmd = row[name]
            # only a cell whose value is stored in files has file_urls
            if cellmd is not None and cellmd.get('file_urls') is not None:
                for url in cellmd['file_urls']:
                    self._add_media_ref(url)
        self._batch.append(row)
        if len(self._batch) >= _ROW_BATCH_SIZE:
            self._write_batch()
        if (
            self._chunk.num_row_bytes >= _MAX_CHUNK_ROW_BYTES
            or self._chunk.num_media_bytes >= _MAX_CHUNK_MEDIA_BYTES
            or len(self._chunk.new_objects) >= _MAX_CHUNK_OBJECTS
        ):
            self._finish_chunk()

    def finish(self) -> None:
        if self._chunk is not None:
            self._finish_chunk()

    def _add_media_ref(self, url: str) -> None:
        assert self._chunk is not None
        path = parse_local_file_path(url)
        # The source table deletes the objects under its table prefix when it is dropped or reverted, so the copy
        # needs its own copy of those. Any other remote url stays as it is.
        if path is None and not ObjectPath.is_table_object(url, self._tbl_id):
            return
        if url in self._exported_media or url in self._chunk.new_media or url in self._chunk.new_objects:
            return
        if path is None:
            # downloaded only for the upload, outside of the export's transaction
            self._chunk.new_objects.add(url)
            return
        if not path.is_file():
            if len(self.missing_files) < _MAX_LISTED_FILES:
                self.missing_files.append(url)
            return
        self._chunk.new_media[url] = str(path)
        self._chunk.num_media_bytes += path.stat().st_size

    def _write_batch(self) -> None:
        assert self._chunk is not None
        if len(self._batch) == 0:
            return
        arrow_cols = {
            col.name: [_to_arrow_value(row[col.name], col.type) for row in self._batch] for col in self._sa_cols
        }
        pa_table = pa.Table.from_pydict(arrow_cols, schema=self._arrow_schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._chunk.rows_file, self._arrow_schema, compression='zstd')
        self._writer.write_table(pa_table)
        self._chunk.num_row_bytes += pa_table.nbytes
        self._batch = []

    def _finish_chunk(self) -> None:
        assert self._chunk is not None
        self._write_batch()
        assert self._writer is not None
        # the file is incomplete until close() writes its footer
        self._writer.close()
        self._exported_media.update(self._chunk.new_media, self._chunk.new_objects)
        self.chunks.append(self._chunk)
        self._chunk = None
        self._writer = None


def _queried_tbl_ids(expr: exprs.Expr) -> set[uuid.UUID]:
    """The ids of the tables that the @pxt.query calls in expr read, at any depth."""
    # referenced_tbl_keys() contains the FROM tables, which a template such as t.limit(n) reads with no ColumnRef. It
    # visits only the clause roots, so the subexprs of each clause are visited as well, for a @pxt.query that a clause
    # calls.
    ids: set[uuid.UUID] = set()
    for e in expr.subexprs():
        if isinstance(e, exprs.FunctionCall) and isinstance(e.fn, func.QueryTemplateFunction):
            template_query = e.fn.template_query
            assert template_query is not None
            ids |= {key.tbl_id for key in template_query.referenced_tbl_keys()}
            for clause_expr in template_query._component_exprs():
                ids |= exprs.Expr.list_tbl_ids(clause_expr.subexprs()) | _queried_tbl_ids(clause_expr)
    return ids


def _check_no_other_tbl_refs(tbl: InsertableTable, md: list[TableVersionMd]) -> None:
    """Refuse tbl if a computed column in md, a list of its versions, references another table. A dropped column
    counts: a revert of a copy restores it."""
    cat = get_runtime().catalog
    looked_up_ids = {tbl._id}
    for version_md in md:
        for col in version_md.schema_version_md.columns.values():
            # The stored expression is searched for the ids of tables. Expr.from_dict() cannot be used: an earlier
            # schema version can reference a column that was dropped since, which it cannot resolve.
            for node in _nested_values(col.value_expr):
                if not isinstance(node, str):
                    continue
                try:
                    other_id = uuid.UUID(node)
                except ValueError:
                    continue
                if other_id in looked_up_ids:
                    continue
                looked_up_ids.add(other_id)
                other = cat.get_table_by_id(other_id)
                if other is not None:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'Cannot copy {tbl._display_str()}: the computed column {col.name!r} references '
                        f'{other._display_str()}; only a table whose computed columns, including dropped ones, '
                        'reference no other table can be copied to another database',
                    )


def _export_table(tbl: InsertableTable, tmp_dir: FsPath, is_other_db: bool) -> _TableExport:
    """Export tbl into tmp_dir, for a copy in another database if is_other_db."""
    cat = get_runtime().catalog
    # a single transaction (REPEATABLE READ) makes the metadata and the rows a snapshot of the same table state
    with cat.begin_data_read_xact(tbl_keys=[TableVersionKey(tbl._id, None)]) as conn:
        # collect_tbl_history() starts a transaction of its own, and the history must come from this one
        md = cat._collect_tbl_history(tbl._id, None)
        tv = tbl._tbl_version.get()
        if is_other_db:
            _check_no_other_tbl_refs(tbl, md)
        store = tv.store_tbl
        export = _TableExport(md, store.sa_tbl, tmp_dir)
        # ordering by version means that the first row to reference a media file is from the earliest version that
        # references it; the destination stores the file under that version, so that reverting a later version
        # keeps it
        order_by = [store.v_min_col, *store.rowid_columns()] if tv.is_data_versioned else store.rowid_columns()
        stmt = sql.select(store.sa_tbl).order_by(*order_by).execution_options(yield_per=_ROW_BATCH_SIZE)
        for row in conn.execute(stmt):
            export.add_row(row._asdict())
        export.finish()
    return export


def copy_table(
    tbl: InsertableTable, dest_catalog: Catalog | CatalogProxy, dest_path: Path, if_exists: IfExistsParam
) -> tuple[Table, bool]:
    """Copy the local base table tbl to dest_path in dest_catalog, the catalog of the same database or a hosted one."""
    # 'error' and 'ignore' are decided before the export, so that no rows are copied for nothing.
    # import_table() does the 'replace'.
    existing = dest_catalog.get_table(dest_path, IfNotExistsParam.IGNORE)
    if existing is not None and if_exists == IfExistsParam.IGNORE:
        return existing, False
    if existing is not None and if_exists == IfExistsParam.ERROR:
        raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {dest_path!r} already exists.')
    if existing is not None and existing._id == tbl._id:
        # the 'replace' would drop tbl and its media files before the copy loads them
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'Cannot replace {tbl._display_str()} with a copy of itself'
        )
    tmp_dir = TempStore.create_path()
    tmp_dir.mkdir(parents=True)
    try:
        export = _export_table(tbl, tmp_dir, is_other_db=dest_path.catalog_uri != tbl._path().catalog_uri)
        if len(export.missing_files) > 0:
            listed = '\n'.join(f'  {url}' for url in export.missing_files)
            raise excs.NotFoundError(
                excs.ErrorCode.STORAGE_NOT_FOUND,
                f'Cannot copy {tbl._display_str()}: it references local media files that no longer exist, '
                f'including:\n{listed}',
            )
        # a chunk downloads its bucket objects when import_table() reads it, outside of the export's transaction
        chunks = (
            CopyChunk(
                str(chunk.rows_file), chunk.new_media | {url: _download(url, tmp_dir) for url in chunk.new_objects}
            )
            for chunk in export.chunks
        )
        return dest_catalog.import_table(dest_path, uuid.uuid4(), export.md, chunks, if_exists)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def create_table_like(
    tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path, if_exists: IfExistsParam
) -> tuple[Table, bool]:
    """Create an empty table at dest_path in dest_catalog with the columns, indexes and settings of the local base table
    tbl, as CREATE TABLE ... LIKE does in SQL."""
    tv = tbl._tbl_version.get()
    is_other_db = dest_path.catalog_uri != tbl._path().catalog_uri
    if is_other_db:
        # only the current version is checked: the new table takes no history
        _check_no_other_tbl_refs(tbl, get_runtime().catalog.collect_tbl_history(tbl._id, 1))

    columns: dict[str, ColumnSpec] = {}
    for col in tv.cols_by_name.values():
        spec: ColumnSpec = {}
        if col.value_expr is None:
            spec['type'] = col.col_type  # type: ignore[typeddict-item]
        elif tbl._id in _queried_tbl_ids(col.value_expr):
            # the query's template is bound to tbl, so the new table would read tbl instead of itself
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot create a table like {tbl._display_str()}: the computed column {col.name!r} calls a @pxt.query '
                f'that reads {tbl._display_str()} itself.',
            )
        else:
            # A reference to a column of tbl becomes a reference by name, which the new table resolves to its own
            # column. For example, t.s.upper() becomes ColumnRefByName('s').upper().
            own_col_refs = [
                ref for ref in col.value_expr.subexprs(exprs.ColumnRef) if ref.col_md.qcolid.tbl_id == tbl._id
            ]
            by_name = exprs.ExprDict[exprs.Expr](
                (ref, exprs.ColumnRefByName(ref.col_md.name, ref.col_md.col_type)) for ref in own_col_refs
            )
            spec['value'] = col.value_expr.copy().substitute(by_name)
            spec['stored'] = col.stored
        if col.is_pk:
            spec['primary_key'] = True
        if col._media_validation is not None:
            spec['media_validation'] = col._media_validation.name.lower()  # type: ignore[typeddict-item]
        destination = col._explicit_destination
        # Another database cannot reach a directory of this machine. The column takes that database's default.
        is_local_dir = (
            isinstance(destination, (str, FsPath))
            and ObjectPath.parse_object_storage_addr(str(destination), False).storage_target
            == StorageTarget.LOCAL_STORE
        )
        if destination is not None and not (is_other_db and is_local_dir):
            spec['destination'] = destination
        if col.custom_metadata is not None:
            spec['custom_metadata'] = col.custom_metadata
        if col.comment is not None:
            spec['comment'] = col.comment
        columns[col.name] = spec

    idxs: list[IndexDefinition] = []
    for info in tv.idxs_by_name.values():
        idx_col = exprs.ColumnRefByName(info.col.name, info.col.col_type)
        if isinstance(info.idx, index.EmbeddingIndex):
            embeddings = info.idx.embeddings
            idxs.append(
                EmbeddingIndex(
                    column=idx_col,
                    string_embed=embeddings.get(ts.ColumnType.Type.STRING),
                    image_embed=embeddings.get(ts.ColumnType.Type.IMAGE),
                    audio_embed=embeddings.get(ts.ColumnType.Type.AUDIO),
                    video_embed=embeddings.get(ts.ColumnType.Type.VIDEO),
                    document_embed=embeddings.get(ts.ColumnType.Type.DOCUMENT),
                    metric=info.idx.metric.name.lower(),  # type: ignore[arg-type]
                    precision=info.idx.precision.value,
                    name=info.name,
                )
            )
        elif not tv.has_default_idxs:
            # with has_default_idxs, the new table creates its B-tree indexes itself
            idxs.append(BtreeIndex(column=idx_col))

    return dest_catalog.create_from_model(
        path=dest_path,
        columns=columns,
        display_name=tbl._display_str(),
        has_default_idxs=tv.has_default_idxs,
        media_validation=tv.media_validation,
        comment=tv.comment,
        custom_metadata=tv.custom_metadata,
        iterator=None,
        base=None,
        idxs=idxs,
        is_data_versioned=tv.is_data_versioned,
        if_exists=if_exists,
    )


def _download(url: str, tmp_dir: FsPath) -> str:
    path = tmp_dir / f'media_{uuid.uuid4().hex}{PurePosixPath(urllib.parse.urlparse(url).path).suffix}'
    ObjectOps.copy_object_to_local_file(url, path)
    return str(path)


def _iter_rows(rows_file: str, columns: list[str] | None = None) -> Iterable[list[dict[str, Any]]]:
    for batch in pq.ParquetFile(rows_file).iter_batches(batch_size=_ROW_BATCH_SIZE, columns=columns):
        yield batch.to_pylist()


def place_media(md: list[TableVersionMd], rows_file: str, new_media: dict[str, str]) -> dict[str, str]:
    """Store the media files new_media (url -> local path) in the media store of the table whose history is md, each
    under the column and version of the first row in rows_file that references it. Returns url -> stored url."""
    tbl_md = md[0].tbl_md
    cols = _store_columns(md)
    media_cols = [col for col in cols if col.is_media]
    cellmd_cols = [col for col in cols if col.stores_cellmd]
    read_cols = [col.store_name for col in media_cols] + [col.cellmd_store_name for col in cellmd_cols]
    if tbl_md.is_data_versioned:
        read_cols.append('v_min')

    placements: dict[str, tuple[str | None, int, int]] = {}  # url -> (destination, column id, table version)
    for rows in _iter_rows(rows_file, read_cols):
        for row in rows:
            version = row['v_min'] if tbl_md.is_data_versioned else tbl_md.current_version
            for col in media_cols:
                url = row[col.store_name]
                if url in new_media and url not in placements:
                    placements[url] = (col.destination, col.col_id, max(version, col.version_add))
            for col in cellmd_cols:
                cellmd = row[col.cellmd_store_name]
                if cellmd is None:
                    continue
                file_urls = json.loads(cellmd).get('file_urls')
                if file_urls is None:
                    continue
                for url in file_urls:
                    if url in new_media and url not in placements:
                        placement = (Env.get().cell_materialization_dest, col.col_id, max(version, col.version_add))
                        placements[url] = placement

    unreferenced = new_media.keys() - placements.keys()
    if len(unreferenced) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT,
            f'Uploaded media files are not referenced by any row: {sorted(unreferenced)}',
        )

    stores: dict[str | None, ObjectStoreBase] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(_MAX_UPLOAD_THREADS, len(placements))), thread_name_prefix='pxt-copy-media'
    ) as executor:
        futures = {}
        for url, (destination, col_id, version) in placements.items():
            if destination not in stores:
                stores[destination] = ObjectOps.get_store(destination, False)
            store = stores[destination]
            src_path = FsPath(new_media[url])
            file_dest = store.resolve_destination(uuid.UUID(tbl_md.tbl_id), col_id, version, ext=src_path.suffix)
            futures[url] = executor.submit(
                ObjectOps.put_file_resolved, store, src_path, file_dest, TempStore.contains_path(src_path)
            )
        return {url: future.result() for url, future in futures.items()}


def _media_destinations(cols: list[_StoreColumn]) -> set[str | None]:
    """The destinations of the files of a table with the stored columns cols, as in TableVersion.delete_media()."""
    return {col.destination for col in cols} | {Env.get().cell_materialization_dest}


def delete_media(md: list[TableVersionMd]) -> None:
    """Delete the media files of the table whose history is md from each of its destinations."""
    tbl_id = uuid.UUID(md[0].tbl_md.tbl_id)
    for destination in _media_destinations(_store_columns(md)):
        ObjectOps.delete(destination, tbl_id)


def _local_media_dirs(tv: TableVersion, cols: list[_StoreColumn]) -> list[FsPath]:
    """The local directories, resolved, that contain the stored media files of tv."""
    addrs = [
        ObjectPath.parse_object_storage_addr(dest or str(Env.get().media_dir), False)
        for dest in _media_destinations(cols)
    ]
    return [
        (addr.to_path / ObjectPath.table_prefix(tv.id)).resolve()
        for addr in addrs
        if addr.storage_target == StorageTarget.LOCAL_STORE
    ]


def _check_stored_url(tv: TableVersion, url: str, local_media_dirs: list[FsPath]) -> str:
    # A query reads the local file that a cell gives. Without this check, a row could give any file of this machine,
    # such as a credential file, and a query would return its content. A copy within one database passes the check as
    # well: the export sends every local file that a row references, so each local url is replaced with a stored one.
    path = parse_local_file_path(url)
    if path is not None and not any(d in path.resolve().parents for d in local_media_dirs):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{url!r} is not a media file of {tv.name!r}')
    return url


def insert_rows(tv: TableVersion, md: list[TableVersionMd], rows_file: str, media_urls: dict[str, str]) -> None:
    """Insert the store rows in rows_file into the store table of tv, whose history is md. A media url that is a key
    of media_urls is replaced with its value."""
    sa_tbl = tv.store_tbl.sa_tbl
    file_cols = pq.ParquetFile(rows_file).schema_arrow.names
    if set(file_cols) != set(sa_tbl.columns.keys()):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT,
            f'The rows do not match the columns of {tv.name!r}: {sorted(file_cols)} != {sorted(sa_tbl.columns.keys())}',
        )
    cols = _store_columns(md)
    media_cols = {col.store_name for col in cols if col.is_media}
    cellmd_cols = {col.cellmd_store_name for col in cols if col.stores_cellmd}
    local_media_dirs = _local_media_dirs(tv, cols)

    conn = get_runtime().conn
    for rows in _iter_rows(rows_file):
        store_rows: list[dict[str, Any]] = []
        for row in rows:
            # a row beyond tv's history is refused, so that a later insert does not reuse its rowid
            if tv.is_data_versioned and (row['rowid'] >= tv.next_row_id or row['v_min'] > tv.version):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'Row {row["rowid"]} is not a row of the history of {tv.name!r}'
                )
            store_row: dict[str, Any] = {}
            for name, val in row.items():
                store_val = _from_arrow_value(val, sa_tbl.columns[name].type)
                if name in media_cols and val is not None:
                    store_val = _check_stored_url(tv, media_urls.get(store_val, store_val), local_media_dirs)
                elif name in cellmd_cols and val is not None and 'file_urls' in store_val:
                    store_val['file_urls'] = [
                        _check_stored_url(tv, media_urls.get(url, url), local_media_dirs)
                        for url in store_val['file_urls']
                    ]
                store_row[name] = store_val
            store_rows.append(store_row)
        conn.execute(sql.insert(sa_tbl), store_rows)

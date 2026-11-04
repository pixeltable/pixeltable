"""Copy a local base table, with its full version history and stored values, into a hosted catalog.

The copy is physical: the table's metadata records and the rows of its store table travel as stored, so no computed
value or embedding is computed again. The destination gives the table a new id, and each local media file that a row
references is uploaded and rewritten to the destination's media store.

copy_table() drives a copy from the source process with existing catalog operations, plus
create_table_from_md(): it exports the table into chunks (a Parquet file of store rows, plus the local media files
those rows are the first to reference), creates the copy under a temporary name, loads each chunk into it with
load_store_rows(), and moves it to its path. A failed copy drops the temporary table.

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
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pgvector.sqlalchemy as sql_vector
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import JSONB

from pixeltable import exceptions as excs, exprs, func, index, type_system as ts
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
    from .catalog_base import CatalogBase
    from .path import Path
    from .table import Table
    from .table_version import TableVersion

_logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 1_000
# A chunk is one load_store_rows() request. These bounds keep each request within the destination's memory, temp disk
# and statement timeout.
_MAX_CHUNK_ROW_BYTES = 64 * 2**20
_MAX_CHUNK_MEDIA_BYTES = 512 * 2**20
# The export bounds the number of bucket objects in a chunk, since it cannot know an object's size before it downloads
# the object, after the export's transaction.
_MAX_CHUNK_OBJECTS = 64
_MAX_UPLOAD_THREADS = 16
# the number of missing media files an error lists
_MAX_LISTED_FILES = 5
# the TableMd.additional_md key that marks a table that create_table_from_md() created and finish_copy() did not finish
_COPY_IN_PROGRESS = 'copy_in_progress'


@dataclasses.dataclass(frozen=True)
class _StoreColumn:
    """A stored column of a table. A dropped column is included: its store column still holds older row versions."""

    col_id: int
    is_media: bool
    is_computed: bool
    stores_cellmd: bool
    # the table version that added the column; adding a computed column fills the existing rows with files of it
    version_add: int

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
    return [
        _StoreColumn(
            col_id=col_id,
            is_media=ts.ColumnType.from_dict(schema_cols[col_id].col_type).is_media_type(),
            is_computed=schema_cols[col_id].value_expr is not None,
            stores_cellmd=col_md.stores_cellmd,
            version_add=min(
                m.version_md.version for m in md if m.version_md.schema_version >= col_md.schema_version_add
            ),
        )
        for col_id, col_md in md[0].tbl_md.column_md.items()
        if col_md.stored
    ]


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


def md_for_copy(
    md: list[TableVersionMd], tbl_id: uuid.UUID, name: str
) -> tuple[schema.TableMd, list[schema.VersionMd], list[schema.SchemaVersionMd]]:
    """The metadata records of a copy, with id tbl_id and named name, of the base table whose history is md."""
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
    # an embedding index builds its functions when the table is loaded, so check those before writing anything
    for idx_md in src_md.index_md.values():
        if not idx_md.is_visible_in_version(src_md.current_schema_version):
            continue
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

    src_id = uuid.UUID(src_md.tbl_id)
    copy_md = [TableVersionMd.from_dict(_remap_tbl_id(m.as_dict(), src_id, tbl_id)) for m in md]
    tbl_md = copy_md[0].tbl_md
    tbl_md.name = name
    tbl_md.user = Env.get().user
    tbl_md.additional_md[_COPY_IN_PROGRESS] = True
    for col_md in tbl_md.column_md.values():
        if (
            isinstance(col_md.destination, str)
            and ObjectPath.parse_object_storage_addr(col_md.destination, False).storage_target
            == StorageTarget.LOCAL_STORE
        ):
            # the source's local directory doesn't exist here; the column falls back to the default destination
            col_md.destination = None
    schema_version_mds = {m.schema_version_md.schema_version: m.schema_version_md for m in copy_md}
    return tbl_md, [m.version_md for m in copy_md], list(schema_version_mds.values())


def check_computed_columns(tv: TableVersion, src_name: str) -> None:
    """Refuse a copy of src_name whose computed columns the destination cannot evaluate."""
    for col in tv.cols_by_id.values():
        validation_error = None if col.value_expr is None else col.value_expr.validation_error
        if validation_error is not None:
            raise excs.NotFoundError(
                excs.ErrorCode.FUNCTION_NOT_FOUND,
                f'Cannot copy {src_name!r}: the computed column {col.name!r} cannot be evaluated in the destination '
                f'database.\n{validation_error.catalog_error_msg()}\n'
                'You can use `pxt db update` to deploy the UDF to the database.',
            )


def is_copy_in_progress(tbl_md: schema.TableMd) -> bool:
    """Whether tbl_md is the metadata of a copy that create_table_from_md() created and finish_copy() did not finish.

    Only load_store_rows() and finish_copy() write to such a table, so that a finished copy holds exactly the rows of
    the copied table.
    """
    return tbl_md.additional_md.get(_COPY_IN_PROGRESS, False)


def _check_copy_in_progress(tbl_id: uuid.UUID, method: str) -> schema.TableMd:
    """Return the TableMd of tbl_id, read from the store (a cached one may be stale), if it is a copy in progress."""
    tbl_md = get_runtime().catalog.load_tbl_md(TableVersionKey(tbl_id, None)).tbl_md
    if not is_copy_in_progress(tbl_md):
        raise excs.RequestError(excs.ErrorCode.INVALID_STATE, f'{method}(): {tbl_md.name!r} is not a copy in progress')
    return tbl_md


def _arrow_type(sa_type: sql.types.TypeEngine) -> pa.DataType:
    # check JSONB before String: the store's types are those of ts._SA_TYPE_NAMES
    if isinstance(sa_type, JSONB):
        return pa.string()  # json text
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
        # a half-precision value is exactly representable as float32
        return np.asarray(val, dtype=np.float32)
    return val


def _from_arrow_value(val: Any, sa_type: sql.types.TypeEngine) -> Any:
    if val is None:
        # the store keeps a SQL NULL in a JSONB column, not a json 'null' (see RowBuilder.create_store_table_row())
        return sql.null() if isinstance(sa_type, JSONB) else None
    if isinstance(sa_type, JSONB):
        return json.loads(val)
    if isinstance(sa_type, sql.UUID):
        return uuid.UUID(bytes=val)
    if isinstance(sa_type, (sql_vector.Vector, sql_vector.HALFVEC)):
        return np.array(val, dtype=np.float32)
    return val


@dataclasses.dataclass
class _Chunk:
    rows_file: FsPath
    num_rows: int = 0
    num_row_bytes: int = 0
    num_media_bytes: int = 0
    # local media files that this chunk is the first to reference: url -> local path
    new_media: dict[str, str] = dataclasses.field(default_factory=dict)
    # urls of the source table's bucket objects that this chunk is the first to reference; they are downloaded for the
    # chunk's upload, and _MAX_CHUNK_OBJECTS bounds their number
    new_objects: set[str] = dataclasses.field(default_factory=set)
    # referenced urls of media files that an earlier chunk uploaded
    prev_media: set[str] = dataclasses.field(default_factory=set)


class _TableExport:
    """Writes the store rows of a table into chunks and collects the media files that they reference and that the copy
    must store itself: local files, and objects that the source table stores under its table prefix."""

    md: list[TableVersionMd]
    chunks: list[_Chunk]
    num_rows: int
    missing_files: list[str]  # urls of local media files that don't exist

    _tmp_dir: FsPath
    _tbl_prefix: str  # the source table's prefix in a media store
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
        self.num_rows = 0
        self.missing_files = []
        self._tmp_dir = tmp_dir
        self._tbl_prefix = ObjectPath.table_prefix(uuid.UUID(md[0].tbl_md.tbl_id))
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
            if cellmd is not None:
                for url in cellmd.get('file_urls') or ():
                    self._add_media_ref(url)
        self._batch.append(row)
        self._chunk.num_rows += 1
        self.num_rows += 1
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
        if path is None and not self._is_source_object(url):
            return  # a remote url that the source table doesn't own is copied as is
        if url in self._exported_media:
            self._chunk.prev_media.add(url)
            return
        if url in self._chunk.new_media or url in self._chunk.new_objects:
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

    def _is_source_object(self, url: str) -> bool:
        """Whether url names an object that the source table stores under its table prefix, which the source's
        drop_table() and revert() delete (ObjectPath.create_prefix_raw() names it <prefix>/../<prefix>_...)."""
        url_path = PurePosixPath(urllib.parse.urlparse(url).path)
        return self._tbl_prefix in url_path.parts and url_path.name.startswith(f'{self._tbl_prefix}_')

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


def _referenced_tbl_ids(expr: exprs.Expr) -> set[uuid.UUID]:
    """The ids of the tables that expr references, including through a @pxt.query at any depth."""
    return exprs.Expr.list_tbl_ids(expr.subexprs()) | _queried_tbl_ids(expr)


def _queried_tbl_ids(expr: exprs.Expr) -> set[uuid.UUID]:
    """The ids of the tables that the @pxt.query calls in expr read, at any depth."""
    # referenced_tbl_keys() holds the FROM tables, which a template such as t.limit(n) reads with no ColumnRef. It
    # visits only the clause roots, so the subexprs of each clause are visited as well, for a @pxt.query that a clause
    # calls.
    ids: set[uuid.UUID] = set()
    for e in expr.subexprs():
        if isinstance(e, exprs.FunctionCall) and isinstance(e.fn, func.QueryTemplateFunction):
            template_query = e.fn.template_query
            assert template_query is not None
            ids |= {key.tbl_id for key in template_query.referenced_tbl_keys()}
            for clause_expr in template_query._component_exprs():
                ids |= _referenced_tbl_ids(clause_expr)
    return ids


def _check_no_other_tbl_refs(tbl: InsertableTable, tv: TableVersion) -> None:
    """Refuse tbl if a computed column of it references another table, which another database cannot resolve."""
    for col in tv.cols_by_id.values():
        if col.value_expr is None:
            continue
        other_ids = _referenced_tbl_ids(col.value_expr) - {tbl._id}
        if len(other_ids) > 0:
            other_id = next(iter(other_ids))
            other = get_runtime().catalog.get_table_by_id(other_id)
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot copy {tbl._display_str()}: the computed column {col.name!r} references '
                f'{other._display_str() if other is not None else other_id}; only a table whose computed '
                'columns reference no other table can be copied to another database',
            )


def _export_table(tbl: InsertableTable, tmp_dir: FsPath) -> _TableExport:
    cat = get_runtime().catalog
    # a single transaction (REPEATABLE READ) makes the metadata and the rows a snapshot of the same table state
    with cat.begin_data_read_xact(tbl_keys=[TableVersionKey(tbl._id, None)]) as conn:
        md = cat._collect_tbl_history(tbl._id, None)
        tv = tbl._tbl_version.get()
        _check_no_other_tbl_refs(tbl, tv)
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
    tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path, if_exists: IfExistsParam
) -> tuple[Table, bool]:
    """Copy the local base table tbl to dest_path in dest_catalog, a hosted catalog.

    Returns (table, was_created).
    """
    from pixeltable.service.proxy_protocol import LocalFile

    from .table_proxy import TableProxy

    # 'error' and 'ignore' are decided before the export, so that no rows are copied for nothing. finish_copy() does
    # the 'replace'.
    existing = dest_catalog.get_table(dest_path, IfNotExistsParam.IGNORE)
    if existing is not None and if_exists == IfExistsParam.IGNORE:
        return existing, False
    if existing is not None and if_exists == IfExistsParam.ERROR:
        raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {dest_path!r} already exists.')
    tmp_dir = TempStore.create_path()
    tmp_dir.mkdir(parents=True)
    try:
        export = _export_table(tbl, tmp_dir)
        if len(export.missing_files) > 0:
            listed = '\n'.join(f'  {url}' for url in export.missing_files)
            raise excs.NotFoundError(
                excs.ErrorCode.STORAGE_NOT_FOUND,
                f'Cannot copy {tbl._display_str()}: it references local media files that no longer exist, '
                f'including:\n{listed}',
            )

        # the copy is filled under a temporary name, so that dest_path only ever names a complete copy
        tmp_path = dest_path.parent.append(f'{dest_path.name}_copy_{uuid.uuid4().hex[:8]}')
        # a failed copy drops tmp_path, so it must not name a table that this copy didn't create
        if dest_catalog.get_table(tmp_path, IfNotExistsParam.IGNORE) is not None:
            raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {tmp_path!r} already exists.')
        try:
            # inside the try: the destination may have created the table even if its response is lost
            copy = dest_catalog.create_table_from_md(tmp_path, export.md)
            assert isinstance(copy, TableProxy)
            uploaded: dict[str, str] = {}  # source url -> its url in the destination
            for i, chunk in enumerate(export.chunks):
                _logger.debug(f'Copying rows chunk {i + 1}/{len(export.chunks)} ({chunk.num_rows} rows)')
                downloads = {url: _download(url, tmp_dir) for url in chunk.new_objects}
                result = copy._dispatch(
                    'load_store_rows',
                    {
                        'rows_file': LocalFile(str(chunk.rows_file)),
                        'new_media': {url: LocalFile(path) for url, path in (chunk.new_media | downloads).items()},
                        'prev_media': {url: uploaded[url] for url in chunk.prev_media},
                    },
                )
                uploaded.update(result)
                for path in downloads.values():
                    FsPath(path).unlink()
            is_moved = copy._dispatch(
                'finish_copy', {'num_rows': export.num_rows, 'dest_path': dest_path, 'if_exists': if_exists}
            )
        except BaseException as e:
            try:
                dest_catalog.drop_table(tmp_path, IfNotExistsParam.IGNORE, force=False)
            except Exception:
                _logger.warning(f'Failed to drop the incomplete copy {tmp_path!r}', exc_info=True)
                e.add_note(f'The incomplete copy {tmp_path!r} remains; drop it with pxt.drop_table().')
            raise
        if not is_moved:
            # if_exists='ignore', and a table was created at dest_path during the copy
            dest_catalog.drop_table(tmp_path, IfNotExistsParam.IGNORE, force=False)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    result_tbl = dest_catalog.get_table(dest_path, IfNotExistsParam.ERROR)
    assert result_tbl is not None
    return result_tbl, is_moved


def create_table_like(
    tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path, if_exists: IfExistsParam
) -> tuple[Table, bool]:
    """Create an empty table at dest_path in dest_catalog with the columns, indexes and settings of the local base table
    tbl, as CREATE TABLE ... LIKE does in SQL.

    Returns (table, was_created).
    """
    tv = tbl._tbl_version.get()
    is_other_db = dest_path.catalog_uri != tbl._path().catalog_uri
    if is_other_db:
        _check_no_other_tbl_refs(tbl, tv)

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


def load_store_rows(
    tbl: InsertableTable, rows_file: str, new_media: dict[str, str], prev_media: dict[str, str]
) -> dict[str, str]:
    """Insert the store rows in the Parquet file rows_file into tbl, a copy in progress (see is_copy_in_progress()).

    Args:
        new_media: url -> local path of each media file that these rows are the first to reference
        prev_media: url -> stored url of each media file that an earlier call stored and these rows reference

    Returns:
        url -> stored url, for each url in new_media
    """
    cat = get_runtime().catalog
    # checked here as well, so that a refused call stores no media files
    with cat.begin_md_read_xact():
        _check_copy_in_progress(tbl._id, 'load_store_rows')
    md = cat.collect_tbl_history(tbl._id, None)
    # object store I/O runs outside of a transaction, so that it doesn't hold a db connection
    placed_media = _place_media(tbl._tbl_version.get(), md, rows_file, new_media)

    with cat.begin_data_write_xact(tvps=[tbl._tbl_version_path]):
        _check_copy_in_progress(tbl._id, 'load_store_rows')
        tv = tbl._tbl_version.get()
        # the client retries a request that gets a 5xx, so this call may repeat one that committed; a call's rows commit
        # together, so its first row tells (an index lookup: the system columns are indexed)
        sa_tbl = tv.store_tbl.sa_tbl
        key_cols = [col.name for col in tv.store_tbl.pk_columns()]
        first_rows = next(iter(_iter_rows(rows_file, key_cols)), [])
        if len(first_rows) > 0:
            key = (sa_tbl.c[name] == _from_arrow_value(first_rows[0][name], sa_tbl.c[name].type) for name in key_cols)
            stmt = sql.select(sql.literal(1)).where(*key)
            if get_runtime().conn.execute(stmt.limit(1)).first() is not None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_STATE,
                    f'load_store_rows(): {tv.name!r} already holds these rows; a call was repeated',
                )
        _insert_rows(tv, md, rows_file, prev_media | placed_media)
    return placed_media


def finish_copy(tbl: InsertableTable, num_rows: int, dest_path: Path, if_exists: IfExistsParam) -> bool:
    """Finish tbl, a copy in progress that load_store_rows() filled: check that it holds num_rows rows, clear its mark
    and move it to dest_path, in one transaction, so that no other write reaches it before it is at dest_path.

    The same transaction applies if_exists to a table at dest_path, so that a failed copy leaves that table as it was.

    Returns whether tbl is at dest_path. It is not when if_exists is 'ignore' and a table is at dest_path.
    """
    from .catalog import retry_md_write_loop

    cat = get_runtime().catalog
    tbl_path = tbl._path()
    key = TableVersionKey(tbl._id, None)
    is_replace = if_exists in (IfExistsParam.REPLACE, IfExistsParam.REPLACE_FORCE)

    @retry_md_write_loop(paths=[tbl_path, dest_path], lock_path_subtree=is_replace)
    def finish() -> bool:
        dest_obj = cat._get_schema_object(dest_path)
        if dest_obj is not None and dest_obj._id == tbl._id:
            return True  # the client repeats a call that committed (it retries a 5xx)
        tbl_md = _check_copy_in_progress(tbl._id, 'finish_copy')
        stmt = sql.select(sql.func.count()).select_from(tbl._tbl_version.get().store_tbl.sa_tbl)
        num_stored_rows = get_runtime().conn.execute(stmt).scalar_one()
        if num_stored_rows != num_rows:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                f'finish_copy(): {tbl_md.name!r} holds {num_stored_rows} rows, not the {num_rows} rows of the copied '
                'table',
            )
        if cat._handle_path_collision(dest_path, InsertableTable, False, if_exists) is not None:
            return False
        del tbl_md.additional_md[_COPY_IN_PROGRESS]
        cat.write_tbl_md(tbl._id, None, tbl_md, None, None)
        cat._clear_tv_cache(key)
        cat._move(tbl_path, dest_path, IfExistsParam.ERROR, IfNotExistsParam.ERROR)
        return True

    return finish()


def _iter_rows(rows_file: str, columns: list[str] | None = None) -> Iterable[list[dict[str, Any]]]:
    for batch in pq.ParquetFile(rows_file).iter_batches(batch_size=_ROW_BATCH_SIZE, columns=columns):
        yield batch.to_pylist()


def _media_destination(tv: TableVersion, col: _StoreColumn) -> str | None:
    if col.col_id in tv.cols_by_id:
        return tv.cols_by_id[col.col_id].destination
    # a dropped column: Column.destination's default for a stored media column
    return Env.get().default_output_media_dest if col.is_computed else Env.get().default_input_media_dest


def _place_media(
    tv: TableVersion, md: list[TableVersionMd], rows_file: str, new_media: dict[str, str]
) -> dict[str, str]:
    """Move the uploaded media files new_media (url -> local path) into the media store of tv.

    Each file is stored under the column and version of the first row in rows_file that references it.

    Returns:
        url -> the url of the stored file, for each url in new_media
    """
    cols = _store_columns(md)
    media_cols = [col for col in cols if col.is_media]
    cellmd_cols = [col for col in cols if col.stores_cellmd]
    read_cols = [col.store_name for col in media_cols] + [col.cellmd_store_name for col in cellmd_cols]
    if tv.is_data_versioned:
        read_cols.append('v_min')

    placements: dict[str, tuple[str | None, int, int]] = {}  # url -> (destination, column id, table version)
    for rows in _iter_rows(rows_file, read_cols):
        for row in rows:
            version = row['v_min'] if tv.is_data_versioned else tv.version
            for col in media_cols:
                url = row[col.store_name]
                if url in new_media and url not in placements:
                    placements[url] = (_media_destination(tv, col), col.col_id, max(version, col.version_add))
            for col in cellmd_cols:
                cellmd = row[col.cellmd_store_name]
                if cellmd is None:
                    continue
                for url in json.loads(cellmd).get('file_urls') or ():
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
            file_dest = store.resolve_destination(tv.id, col_id, version, ext=src_path.suffix)
            futures[url] = executor.submit(
                ObjectOps.put_file_resolved, store, src_path, file_dest, TempStore.contains_path(src_path)
            )
        return {url: future.result() for url, future in futures.items()}


def _local_media_dirs(tv: TableVersion, cols: list[_StoreColumn]) -> list[FsPath]:
    """The local directories, resolved, in which _place_media() stores the media files of tv."""
    dests = {_media_destination(tv, col) for col in cols if col.is_media} | {Env.get().cell_materialization_dest}
    addrs = [ObjectPath.parse_object_storage_addr(dest or str(Env.get().media_dir), False) for dest in dests]
    return [
        (addr.to_path / ObjectPath.table_prefix(tv.id)).resolve()
        for addr in addrs
        if addr.storage_target == StorageTarget.LOCAL_STORE
    ]


def _check_stored_url(tv: TableVersion, url: str, local_media_dirs: list[FsPath]) -> str:
    # the destination reads a local file that a cell names, so a caller must not name one of the destination's own
    # files, such as a credential
    path = parse_local_file_path(url)
    if path is not None and not any(d in path.resolve().parents for d in local_media_dirs):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'load_store_rows(): {url!r} is not a media file of {tv.name!r}'
        )
    return url


def _insert_rows(tv: TableVersion, md: list[TableVersionMd], rows_file: str, media_urls: dict[str, str]) -> None:
    """Insert the store rows in rows_file into the store table of tv, rewriting each media url in media_urls.

    In a data-versioned table, every row must be a row of tv's history: its rowid below tv's next rowid and its v_min
    at most tv's version, so that a later insert does not reuse its rowid. Every local media url that the rows end up
    with must name a file in a media directory of tv.
    """
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
            if tv.is_data_versioned and (row['rowid'] >= tv.next_row_id or row['v_min'] > tv.version):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'load_store_rows(): row {row["rowid"]} is not a row of the history of {tv.name!r}',
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

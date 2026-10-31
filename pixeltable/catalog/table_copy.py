"""Copy a local base table, with its full version history and stored values, into a hosted catalog.

The copy is physical: the table's metadata records and the rows of its store table travel as stored, so no computed
value or embedding is computed again. The destination gives the table a new id, and each local media file that a row
references is uploaded and rewritten to the destination's media store.

copy_table() drives a copy from the source process with existing catalog operations, plus
create_table_from_md(): it exports the table into chunks (a Parquet file of store rows, plus the local media files
those rows are the first to reference), creates the copy under a temporary name, loads each chunk into it with
load_store_rows(), and moves it to its path. A failed copy drops the temporary table.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as FsPath
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pgvector.sqlalchemy as sql_vector
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import JSONB

from pixeltable import exceptions as excs, func, type_system as ts
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, ObjectStoreBase, StorageTarget

from .column import Column
from .globals import IfExistsParam, IfNotExistsParam
from .types import TableVersionMd

if TYPE_CHECKING:
    from .catalog_base import CatalogBase
    from .insertable_table import InsertableTable
    from .path import Path
    from .table import Table
    from .table_version import TableVersion

_logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 1_000
# A chunk is one load_store_rows() request. These bounds keep each request within the destination's memory, temp disk
# and statement timeout.
_MAX_CHUNK_ROW_BYTES = 64 * 2**20
_MAX_CHUNK_MEDIA_BYTES = 512 * 2**20
_MAX_UPLOAD_THREADS = 16
# the number of missing media files an error lists
_MAX_LISTED_FILES = 5


@dataclasses.dataclass(frozen=True)
class _StoreColumn:
    """A stored column of a table. A dropped column is included: its store column still holds older row versions."""

    col_id: int
    is_media: bool
    is_computed: bool
    stores_cellmd: bool

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
    # referenced urls of local media files that an earlier chunk uploaded
    prev_media: set[str] = dataclasses.field(default_factory=set)


class _TableExport:
    """Writes the store rows of a table into chunks and collects the local media files that they reference."""

    md: list[TableVersionMd]
    chunks: list[_Chunk]
    num_rows: int
    missing_files: list[str]  # urls of local media files that don't exist

    _tmp_dir: FsPath
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
        if self._chunk.num_row_bytes >= _MAX_CHUNK_ROW_BYTES or self._chunk.num_media_bytes >= _MAX_CHUNK_MEDIA_BYTES:
            self._finish_chunk()

    def finish(self) -> None:
        if self._chunk is not None:
            self._finish_chunk()

    def _add_media_ref(self, url: str) -> None:
        assert self._chunk is not None
        path = parse_local_file_path(url)
        if path is None:
            return  # a remote url is copied as is
        if url in self._exported_media:
            self._chunk.prev_media.add(url)
            return
        if url in self._chunk.new_media:
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
        self._exported_media.update(self._chunk.new_media)
        self.chunks.append(self._chunk)
        self._chunk = None
        self._writer = None


def _export_table(tbl: InsertableTable, tmp_dir: FsPath) -> _TableExport:
    cat = get_runtime().catalog
    # a single REPEATABLE READ transaction makes the metadata and the rows a snapshot of the same table state
    with cat.begin_xact(for_write=False, read_tbl_ids=[tbl._id], isolation_level='REPEATABLE READ') as conn:
        md = cat._collect_tbl_history(tbl._id, None)
        tv = tbl._tbl_version.get()
        # the destination cannot resolve a reference to another local table, such as a @pxt.query UDF's
        for col in tv.cols_by_id.values():
            if col.value_expr is None:
                continue
            # FunctionCall.tbl_ids() adds a @pxt.query's tables only for the call itself, so visit every subexpr
            other_ids = {tbl_id for e in col.value_expr.subexprs() for tbl_id in e.tbl_ids()} - {tbl._id}
            if len(other_ids) > 0:
                other_id = next(iter(other_ids))
                other = cat.get_table_by_id(other_id)
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Cannot copy {tbl._display_str()}: the computed column {col.name!r} references '
                    f'{other._display_str() if other is not None else other_id}; only a table whose computed '
                    'columns reference no other table can be copied',
                )
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


def copy_table(tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path) -> Table:
    """Copy the local base table tbl to dest_path in dest_catalog, a hosted catalog."""
    from pixeltable.service.proxy_protocol import LocalFile

    from .table_proxy import TableProxy

    # fail before the export, rather than at the move that ends the copy
    if dest_catalog.get_table(dest_path, IfNotExistsParam.IGNORE) is not None:
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
            uploaded: dict[str, str] = {}  # local url -> its url in the destination
            num_rows = 0
            for i, chunk in enumerate(export.chunks):
                _logger.debug(f'Copying rows chunk {i + 1}/{len(export.chunks)} ({chunk.num_rows} rows)')
                result = copy._dispatch(
                    'load_store_rows',
                    {
                        'rows_file': LocalFile(str(chunk.rows_file)),
                        'new_media': {url: LocalFile(path) for url, path in chunk.new_media.items()},
                        'prev_media': {url: uploaded[url] for url in chunk.prev_media},
                        'num_prev_rows': num_rows,
                    },
                )
                uploaded.update(result['placed_media'])
                num_rows += result['num_rows']
            if num_rows != export.num_rows:
                raise excs.InternalError(
                    excs.ErrorCode.INTERNAL_ERROR,
                    f'The copy of {tbl._display_str()} is incomplete: {num_rows} of {export.num_rows} rows were stored',
                )
            dest_catalog.move(tmp_path, dest_path, IfExistsParam.ERROR, IfNotExistsParam.ERROR)
        except BaseException:
            try:
                dest_catalog.drop_table(tmp_path, IfNotExistsParam.IGNORE, force=False)
            except Exception:
                _logger.warning(f'Failed to drop the incomplete copy {tmp_path!r}', exc_info=True)
            raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    result_tbl = dest_catalog.get_table(dest_path, IfNotExistsParam.ERROR)
    assert result_tbl is not None
    return result_tbl


def load_store_rows(
    tbl: InsertableTable, rows_file: str, new_media: dict[str, str], prev_media: dict[str, str], num_prev_rows: int
) -> dict[str, Any]:
    """Insert the store rows in the Parquet file rows_file into tbl, a copy that create_table_from_md() created.

    Args:
        new_media: url -> local path of each media file that these rows are the first to reference
        prev_media: url -> stored url of each media file that an earlier call stored and these rows reference
        num_prev_rows: the number of rows that earlier calls inserted

    Returns:
        {'placed_media': url -> stored url for each url in new_media, 'num_rows': the number of inserted rows}
    """
    from .catalog import retry_loop

    md = get_runtime().catalog.collect_tbl_history(tbl._id, None)
    # object store I/O runs outside of a transaction, so that it doesn't hold a db connection
    placed_media = _place_media(tbl._tbl_version.get(), md, rows_file, new_media)

    @retry_loop(for_write=True, write_tbl_ids=[tbl._id])
    def insert() -> int:
        tv = tbl._tbl_version.get()
        # the client retries a request that gets a 5xx, so this call may repeat one that committed
        stmt = sql.select(sql.func.count()).select_from(tv.store_tbl.sa_tbl)
        num_rows = get_runtime().conn.execute(stmt).scalar_one()
        if num_rows != num_prev_rows:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                f'load_store_rows(): {tv.name!r} holds {num_rows} rows, not the {num_prev_rows} rows of the earlier '
                'calls; a call was repeated or skipped',
            )
        return _insert_rows(tv, md, rows_file, prev_media | placed_media)

    return {'placed_media': placed_media, 'num_rows': insert()}


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
                    placements[url] = (_media_destination(tv, col), col.col_id, version)
            for col in cellmd_cols:
                cellmd = row[col.cellmd_store_name]
                if cellmd is None:
                    continue
                for url in json.loads(cellmd).get('file_urls') or ():
                    if url in new_media and url not in placements:
                        placements[url] = (Env.get().cell_materialization_dest, col.col_id, version)

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


def _insert_rows(tv: TableVersion, md: list[TableVersionMd], rows_file: str, media_urls: dict[str, str]) -> int:
    """Insert the store rows in rows_file into the store table of tv, rewriting each media url in media_urls.

    Every local media url that the rows end up with must name a file in a media directory of tv.

    Returns:
        the number of inserted rows
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
    num_rows = 0
    for rows in _iter_rows(rows_file):
        store_rows: list[dict[str, Any]] = []
        for row in rows:
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
        num_rows += len(store_rows)
    return num_rows

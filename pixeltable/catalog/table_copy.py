"""Copy a local base table, with its full version history and stored values, into a hosted catalog.

The copy is physical: the table's metadata records and the rows of its store table travel as stored, so no computed
value or embedding is computed again. The destination gives the table a new id, and each local media file that a row
references is uploaded and rewritten to the destination's media store.

copy_table() drives a copy from the source process: it exports the table into chunks (a Parquet file of store rows,
plus the local media files those rows are the first to reference), then calls the destination catalog's
begin_table_import(), import_table_rows() for each chunk, and finish_table_import(). place_media() and insert_rows()
are the destination's side of import_table_rows().
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
import pgvector.sqlalchemy as sql_vector  # type: ignore[import-untyped]
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql
from sqlalchemy.dialects.postgresql import JSONB

from pixeltable import exceptions as excs, metadata, type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime
from pixeltable.utils import parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectStoreBase

from .column import Column

if TYPE_CHECKING:
    from .catalog_base import CatalogBase
    from .insertable_table import InsertableTable
    from .path import Path
    from .table import Table
    from .table_version import TableVersion

_logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 1_000
# A chunk is one import_table_rows() request. These bounds keep each request within the destination's memory, temp
# disk and statement timeout.
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


def store_columns(md: Mapping[str, Any]) -> list[_StoreColumn]:
    """The stored columns described by md, as read_md_for_copy() returns it."""
    # the latest schema version that has a column describes it
    schema_cols: dict[int, dict[str, Any]] = {}
    for schema_version_md in sorted(md['schema_version_mds'], key=lambda d: d['schema_version']):
        for col_id, schema_col in schema_version_md['columns'].items():
            schema_cols[int(col_id)] = schema_col
    result: list[_StoreColumn] = []
    for col_id_str, col_md in md['tbl_md']['column_md'].items():
        if not col_md['stored']:
            continue
        schema_col = schema_cols[int(col_id_str)]
        result.append(
            _StoreColumn(
                col_id=int(col_id_str),
                is_media=ts.ColumnType.from_dict(schema_col['col_type']).is_media_type(),
                is_computed=schema_col['value_expr'] is not None,
                stores_cellmd=col_md['stores_cellmd'],
            )
        )
    return result


def remap_tbl_id(md: Any, old_id: uuid.UUID, new_id: uuid.UUID) -> Any:
    """Return md with every reference to old_id (a string value equal to str(old_id)) replaced with new_id."""
    old_str, new_str = str(old_id), str(new_id)
    if isinstance(md, str):
        return new_str if md == old_str else md
    if isinstance(md, list):
        return [remap_tbl_id(x, old_id, new_id) for x in md]
    if isinstance(md, dict):
        return {k: remap_tbl_id(v, old_id, new_id) for k, v in md.items()}
    return md


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
    if isinstance(sa_type, sql_vector.HALFVEC):
        return val.to_numpy().astype(np.float32)
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

    md: dict[str, Any]
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

    def __init__(self, md: dict[str, Any], sa_tbl: sql.Table, tmp_dir: FsPath):
        self.md = md
        self.chunks = []
        self.num_rows = 0
        self.missing_files = []
        self._tmp_dir = tmp_dir
        self._sa_cols = list(sa_tbl.columns)
        self._arrow_schema = pa.schema([(col.name, _arrow_type(col.type)) for col in self._sa_cols])
        cols = store_columns(md)
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
        md = cat.read_md_for_copy(tbl._id)
        tv = tbl._tbl_version.get()
        store = tv.store_tbl
        export = _TableExport(md, store.sa_tbl, tmp_dir)
        # ordering by version means that the first row to reference a media file is from the earliest version that
        # references it; the destination stores the file under that version, so that reverting a later version
        # keeps it
        order_by = [store.v_min_col, *store.rowid_columns()] if tv.is_data_versioned else store.rowid_columns()
        stmt = sql.select(store.sa_tbl).order_by(*order_by).execution_options(yield_per=_ROW_BATCH_SIZE)
        for row in conn.execute(stmt):
            export.add_row(row._mapping)
        export.finish()
    return export


def copy_table(tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path) -> Table:
    """Copy the local base table tbl to dest_path in dest_catalog."""
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

        import_id = dest_catalog.begin_table_import(dest_path, export.md, metadata.VERSION)
        try:
            uploaded: dict[str, str] = {}  # local url -> its url in the destination
            for i, chunk in enumerate(export.chunks):
                _logger.debug(f'Copying rows chunk {i + 1}/{len(export.chunks)} ({chunk.num_rows} rows)')
                placed = dest_catalog.import_table_rows(
                    import_id,
                    str(chunk.rows_file),
                    chunk.new_media,
                    {url: uploaded[url] for url in chunk.prev_media},
                )
                uploaded.update(placed)
            return dest_catalog.finish_table_import(import_id, dest_path, export.num_rows)
        except BaseException:
            try:
                dest_catalog.abort_table_import(import_id)
            except Exception:
                _logger.warning(f'Failed to clean up the incomplete copy of {tbl._display_str()}', exc_info=True)
            raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _iter_rows(rows_file: str, columns: list[str] | None = None) -> Iterable[list[dict[str, Any]]]:
    for batch in pq.ParquetFile(rows_file).iter_batches(batch_size=_ROW_BATCH_SIZE, columns=columns):
        yield batch.to_pylist()


def _media_destination(tv: TableVersion, col: _StoreColumn) -> str | None:
    if col.col_id in tv.cols_by_id:
        return tv.cols_by_id[col.col_id].destination
    # a dropped column: Column.destination's default for a stored media column
    return Env.get().default_output_media_dest if col.is_computed else Env.get().default_input_media_dest


def place_media(tv: TableVersion, md: dict[str, Any], rows_file: str, new_media: dict[str, str]) -> dict[str, str]:
    """Move the uploaded media files new_media (url -> local path) into the media store of tv.

    Each file is stored under the column and version of the first row in rows_file that references it.

    Returns:
        url -> the url of the stored file, for each url in new_media
    """
    cols = store_columns(md)
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
            excs.ErrorCode.INVALID_ARGUMENT, f'Uploaded media files are not referenced by any row: {sorted(unreferenced)}'
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


def insert_rows(tv: TableVersion, md: dict[str, Any], rows_file: str, media_urls: dict[str, str]) -> int:
    """Insert the store rows in rows_file into the store table of tv, rewriting each media url in media_urls.

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
    cols = store_columns(md)
    media_cols = {col.store_name for col in cols if col.is_media}
    cellmd_cols = {col.cellmd_store_name for col in cols if col.stores_cellmd}

    conn = get_runtime().conn
    num_rows = 0
    for rows in _iter_rows(rows_file):
        store_rows: list[dict[str, Any]] = []
        for row in rows:
            store_row: dict[str, Any] = {}
            for name, val in row.items():
                store_val = _from_arrow_value(val, sa_tbl.columns[name].type)
                if name in media_cols and val is not None:
                    store_val = media_urls.get(store_val, store_val)
                elif name in cellmd_cols and val is not None and 'file_urls' in store_val:
                    store_val['file_urls'] = [media_urls.get(url, url) for url in store_val['file_urls']]
                store_row[name] = store_val
            store_rows.append(store_row)
        conn.execute(sql.insert(sa_tbl), store_rows)
        num_rows += len(store_rows)
    return num_rows

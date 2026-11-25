"""Copy the current version of a local base table to a path in the same database or in a hosted database. The copy
keeps the stored values of computed columns and indexes and does not compute them again. The copy has no history: it
starts at version 0.

copy_table() exports the current version of the table into chunks of store rows and media files and passes the chunks
to import_table() of the destination catalog. import_table() creates the table with all of its rows in one
transaction. A copy without rows is an empty table with the columns, indexes and settings of the source table.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
import urllib.parse
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as FsPath, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql

from pixeltable import exceptions as excs, func, type_system as ts
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.runtime import get_runtime
from pixeltable.utils import arrow, parse_local_file_path
from pixeltable.utils.local_store import TempStore
from pixeltable.utils.object_stores import ObjectOps, ObjectPath, ObjectStoreBase, StorageTarget

from .column import Column
from .globals import IfExistsParam, IfNotExistsParam
from .table_version import TableVersion
from .types import TableVersionKey, TableVersionMd

if TYPE_CHECKING:
    from pixeltable.store import StoreBase

    from .catalog_base import CatalogBase
    from .insertable_table import InsertableTable
    from .path import Path
    from .table import Table

_logger = logging.getLogger(__name__)

_ROW_BATCH_SIZE = 1_000
# A chunk is a rows file and the media files that first appear in its rows. The destination downloads and stores one
# chunk at a time. These bounds limit the temp disk space that the destination uses.
_MAX_CHUNK_ROW_BYTES = 64 * 2**20
_MAX_CHUNK_MEDIA_BYTES = 512 * 2**20
# The export cannot know the size of a bucket object, because the object is downloaded after the export's
# transaction. _MAX_CHUNK_OBJECTS limits the number of objects in a chunk instead.
_MAX_CHUNK_OBJECTS = 64
_MAX_UPLOAD_THREADS = 16
# the maximum number of missing media files in an error message
_MAX_LISTED_FILES = 5


def _store_columns(md: TableVersionMd) -> list[Column]:
    """Return the stored columns of the current version of the table with metadata md."""
    store_cols: list[Column] = []
    for col_id, schema_col in md.schema_version_md.columns.items():
        col_md = md.tbl_md.column_md[col_id]
        if not col_md.stored:
            continue
        assert col_md.sa_col_type is not None
        store_cols.append(
            Column(
                schema_col.name,
                col_id=col_id,
                col_type=ts.ColumnType.from_dict(schema_col.col_type),
                sa_col_type=ts.sa_type_from_dict(col_md.sa_col_type),
                stores_cellmd=col_md.stores_cellmd,
                value_expr_dict=schema_col.value_expr,
                destination=col_md.destination,
            )
        )
    return store_cols


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


def _is_local_dir(destination: Any) -> bool:
    """Return True if destination is a directory on this machine."""
    return (
        isinstance(destination, str)
        and ObjectPath.parse_object_storage_addr(destination, False).storage_target == StorageTarget.LOCAL_STORE
    )


def md_for_copy(md: TableVersionMd, tbl_id: uuid.UUID, name: str) -> TableVersionMd:
    """Return the metadata of a copy of the table version md. tbl_id and name are the id and the name of the copy, and
    its only version is 0. Raise an error if the table cannot be copied into this database."""
    src_md = md.tbl_md
    if src_md.view_md is not None:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot copy {src_md.name!r}: only base tables can be copied'
        )
    if src_md.pending_stmt is not None or src_md.tbl_state != schema.TableState.LIVE:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_STATE,
            f'Cannot copy {src_md.name!r} while an operation on it is in progress. Retry the copy later.',
        )
    _check_udfs(md)

    src_id = uuid.UUID(src_md.tbl_id)
    copy_md = TableVersionMd.from_dict(_remap_tbl_id(md.as_dict(), src_id, tbl_id))
    tbl_md = copy_md.tbl_md
    tbl_md.name = name
    tbl_md.user = Env.get().user
    # The dropped columns and indexes of a data-versioned table stay in its metadata, so that a revert can restore
    # them. The copy cannot be reverted and leaves them out.
    tbl_md.column_md = {
        col_id: col_md
        for col_id, col_md in tbl_md.column_md.items()
        if col_md.is_visible_in_version(src_md.current_schema_version)
    }
    tbl_md.index_md = {
        idx_id: idx_md
        for idx_id, idx_md in tbl_md.index_md.items()
        if idx_md.is_visible_in_version(src_md.current_schema_version)
    }
    for col_md in tbl_md.column_md.values():
        col_md.schema_version_add = 0
        if Env.get().is_proxy_daemon and _is_local_dir(col_md.destination):
            # The local directory of a column on the client machine does not exist on the proxy daemon. The column uses
            # the default destination instead.
            col_md.destination = None
    for idx_md in tbl_md.index_md.values():
        idx_md.schema_version_add = 0
    tbl_md.current_version = 0
    tbl_md.current_schema_version = 0
    version_md = copy_md.version_md
    version_md.version = 0
    version_md.schema_version = 0
    version_md.created_at = time.time()
    version_md.user = Env.get().user
    version_md.update_status = None
    copy_md.schema_version_md.schema_version = 0
    copy_md.schema_version_md.preceding_schema_version = None
    return copy_md


def _check_udfs(md: TableVersionMd) -> None:
    """Raise an error if this database cannot load a UDF that an index or a computed column of the table version md
    uses."""
    src_name = md.tbl_md.name
    # (description, serialized index arguments or value expression)
    udf_callers: list[tuple[str, Any]] = [
        (f'the index {idx_md.name!r}', idx_md.init_args)
        for idx_md in md.tbl_md.index_md.values()
        if idx_md.is_visible_in_version(md.tbl_md.current_schema_version)
    ]
    for col in md.schema_version_md.columns.values():
        # the value column of an index has no name
        col_desc = f'the computed column {col.name!r}' if col.name is not None else 'the value column of an index'
        udf_callers.append((col_desc, col.value_expr))
    for desc, caller_md in udf_callers:
        # This loads only the functions, because Expr.from_dict() needs the metadata of the copy, which does not exist
        # yet. A function with a path is a UDF. A @pxt.query has no path, and its UDFs are nested in it.
        for node in _nested_values(caller_md):
            if not (isinstance(node, dict) and '_classpath' in node and 'path' in node):
                continue
            udf = func.Function.from_dict(node)
            if isinstance(udf, func.InvalidFunction):
                raise excs.NotFoundError(
                    excs.ErrorCode.FUNCTION_NOT_FOUND,
                    f'Cannot copy {src_name!r}: {desc} uses the UDF `{udf.self_path}`, which is not defined in the '
                    'destination database.\nYou can use `pxt db update` to deploy the UDF to the database.',
                )


@dataclasses.dataclass(frozen=True)
class CopyChunk:
    """A part of a table copy: the store rows in the Parquet file rows_file, and the media files that first appear in
    these rows (media: url -> local path)."""

    rows_file: str
    media: dict[str, str]


@dataclasses.dataclass
class _Chunk:
    """The store rows and media files of a table export that become one CopyChunk."""

    rows_file: FsPath
    num_row_bytes: int = 0
    num_media_bytes: int = 0
    # local media files that first appear in this chunk: url -> local path
    new_media: dict[str, str] = dataclasses.field(default_factory=dict)
    # urls of the bucket objects of the source table that first appear in this chunk, to download for its CopyChunk
    new_objects: set[str] = dataclasses.field(default_factory=set)


class _TableExport:
    """Writes the store rows of a table into chunks. A chunk also contains the media files to copy: local files and
    the objects under the table prefix of the source table."""

    md: TableVersionMd
    # the store columns that the export reads: the rowid and the columns of the current version, without v_min and v_max
    sa_cols: list[sql.Column]
    chunks: list[_Chunk]
    missing_files: list[str]  # urls of local media files that don't exist

    _tmp_dir: FsPath
    _tbl_id: uuid.UUID
    _arrow_schema: pa.Schema
    _media_cols: list[str]
    _cellmd_cols: list[str]
    _exported_media: set[str]  # urls in the new_media and new_objects of the finished chunks
    _chunk: _Chunk | None
    _writer: pq.ParquetWriter | None
    _batch: list[Mapping[str, Any]]

    def __init__(self, md: TableVersionMd, store: StoreBase, tmp_dir: FsPath):
        self.md = md
        cols = _store_columns(md)
        self.sa_cols = list(store.rowid_columns())
        for col in cols:
            self.sa_cols.append(store.sa_tbl.c[col.store_name()])
            if col.stores_cellmd:
                self.sa_cols.append(store.sa_tbl.c[col.cellmd_store_name()])
        self.chunks = []
        self.missing_files = []
        self._tmp_dir = tmp_dir
        self._tbl_id = uuid.UUID(md.tbl_md.tbl_id)
        self._arrow_schema = pa.schema([(col.name, arrow.sa_to_arrow_type(col.type)) for col in self.sa_cols])
        self._media_cols = [col.store_name() for col in cols if col.col_type.is_media_type()]
        self._cellmd_cols = [col.cellmd_store_name() for col in cols if col.stores_cellmd]
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
        # Dropping or reverting the source table deletes the objects under its table prefix, and the copy needs its own
        # copies of those objects. Any other remote url stays as it is.
        if path is None and not ObjectPath.is_table_object(url, self._tbl_id):
            return
        if url in self._exported_media or url in self._chunk.new_media or url in self._chunk.new_objects:
            return
        if path is None:
            # the object is downloaded later, outside of the export's transaction
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
            col.name: [arrow.sa_val_to_arrow_val(row[col.name], col.type) for row in self._batch]
            for col in self.sa_cols
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


def _check_no_other_tbl_refs(tbl: InsertableTable, md: TableVersionMd) -> None:
    """Raise an error if a computed column of the table version md of tbl references another table."""
    cat = get_runtime().catalog
    looked_up_ids = {tbl._id}
    for col in md.schema_version_md.columns.values():
        # Search the stored expression for table ids. These include the ids of the tables that a @pxt.query reads.
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
                    f'{other._display_str()}. A table can be copied to another database only if its computed columns '
                    'reference no other table.',
                )


def _export_table(tbl: InsertableTable, tmp_dir: FsPath, is_other_db: bool, with_rows: bool) -> _TableExport:
    """Export the current version of tbl: its metadata and, if with_rows is True, its rows as chunks in tmp_dir.
    is_other_db is True if the copy goes to another database."""
    cat = get_runtime().catalog
    # one REPEATABLE READ transaction reads the metadata and the rows, so that both come from the same table state
    with cat.begin_data_read_xact(tbl_keys=[TableVersionKey(tbl._id, None)]) as conn:
        # collect_tbl_history() starts its own transaction, and the metadata must come from this one
        md = cat._collect_tbl_history(tbl._id, 1)[0]
        tv = tbl._tbl_version.get()
        if is_other_db:
            _check_no_other_tbl_refs(tbl, md)
        store = tv.store_tbl
        export = _TableExport(md, store, tmp_dir)
        if with_rows:
            stmt = sql.select(*export.sa_cols).order_by(*store.rowid_columns())
            if tv.is_data_versioned:
                # only the rows of the current version: a row of an earlier version has a smaller v_max
                stmt = stmt.where(store.v_max_col == schema.Table.MAX_VERSION)
            stmt = stmt.execution_options(yield_per=_ROW_BATCH_SIZE)
            for row in conn.execute(stmt):
                export.add_row(row._asdict())
            export.finish()
    return export


def copy_table(
    tbl: InsertableTable, dest_catalog: CatalogBase, dest_path: Path, with_rows: bool, if_exists: IfExistsParam
) -> tuple[Table, bool]:
    """Copy the current version of the local base table tbl to dest_path in dest_catalog. dest_catalog is the catalog
    of the same database or of a hosted database. If with_rows is False, the copy is an empty table with the columns,
    indexes and settings of tbl."""
    # Handle 'error' and 'ignore' before the export, so that a copy that creates no table exports no rows.
    # import_table() handles 'replace'.
    existing = dest_catalog.get_table(dest_path, IfNotExistsParam.IGNORE)
    if existing is not None and if_exists == IfExistsParam.IGNORE:
        # 'ignore' does not return a view, because the new table is a base table
        if existing._tbl_path.is_view():
            raise excs.AlreadyExistsError(
                excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {dest_path!r} already exists and is not a table'
            )
        return existing, False
    if existing is not None and if_exists == IfExistsParam.ERROR:
        raise excs.AlreadyExistsError(excs.ErrorCode.PATH_ALREADY_EXISTS, f'Path {dest_path!r} already exists.')
    if with_rows and existing is not None and existing._id == tbl._id:
        # the 'replace' would drop tbl and its media files before the copy loads them
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'Cannot replace {tbl._display_str()} with a copy of itself'
        )
    tmp_dir = TempStore.create_path()
    tmp_dir.mkdir(parents=True)
    try:
        export = _export_table(
            tbl, tmp_dir, is_other_db=dest_path.catalog_uri != tbl._path().catalog_uri, with_rows=with_rows
        )
        if len(export.missing_files) > 0:
            listed = '\n'.join(f'  {url}' for url in export.missing_files)
            raise excs.NotFoundError(
                excs.ErrorCode.STORAGE_NOT_FOUND,
                f'Cannot copy {tbl._display_str()}: it references local media files that no longer exist, '
                f'including:\n{listed}',
            )
        # the generator downloads the bucket objects of a chunk when import_table() reads the chunk, after the export's
        # transaction
        chunks = (
            CopyChunk(
                str(chunk.rows_file), chunk.new_media | {url: _download(url, tmp_dir) for url in chunk.new_objects}
            )
            for chunk in export.chunks
        )
        return dest_catalog.import_table(dest_path, uuid.uuid4(), export.md, chunks, if_exists)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _download(url: str, tmp_dir: FsPath) -> str:
    path = tmp_dir / f'media_{uuid.uuid4().hex}{PurePosixPath(urllib.parse.urlparse(url).path).suffix}'
    ObjectOps.copy_object_to_local_file(url, path)
    return str(path)


def _iter_rows(rows_file: str, columns: list[str] | None = None) -> Iterable[list[dict[str, Any]]]:
    for batch in pq.ParquetFile(rows_file).iter_batches(batch_size=_ROW_BATCH_SIZE, columns=columns):
        yield batch.to_pylist()


def place_media(md: TableVersionMd, rows_file: str, new_media: dict[str, str]) -> dict[str, str]:
    """Store each media file of new_media (url -> local path) in the media store of the copy with metadata md. A file
    goes to the destination of the column that first references it in rows_file. Return url -> stored url."""
    tbl_md = md.tbl_md
    cols = _store_columns(md)
    media_cols = [col for col in cols if col.col_type.is_media_type()]
    cellmd_cols = [col for col in cols if col.stores_cellmd]
    read_cols = [col.store_name() for col in media_cols] + [col.cellmd_store_name() for col in cellmd_cols]

    placements: dict[str, tuple[str | None, int]] = {}  # url -> (destination, column id)
    for rows in _iter_rows(rows_file, read_cols):
        for row in rows:
            for col in media_cols:
                url = row[col.store_name()]
                if url in new_media and url not in placements:
                    placements[url] = (col.destination, col.id)
            for col in cellmd_cols:
                cellmd = row[col.cellmd_store_name()]
                if cellmd is None:
                    continue
                file_urls = json.loads(cellmd).get('file_urls')
                if file_urls is None:
                    continue
                for url in file_urls:
                    if url in new_media and url not in placements:
                        placements[url] = (Env.get().cell_materialization_dest, col.id)

    unreferenced = new_media.keys() - placements.keys()
    if len(unreferenced) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, f'No row references the uploaded media files {sorted(unreferenced)}'
        )

    stores: dict[str | None, ObjectStoreBase] = {}
    with ThreadPoolExecutor(
        max_workers=max(1, min(_MAX_UPLOAD_THREADS, len(placements))), thread_name_prefix='pxt-copy-media'
    ) as executor:
        futures = {}
        for url, (destination, col_id) in placements.items():
            if destination not in stores:
                stores[destination] = ObjectOps.get_store(destination, False)
            store = stores[destination]
            src_path = FsPath(new_media[url])
            file_dest = store.resolve_destination(
                uuid.UUID(tbl_md.tbl_id), col_id, tbl_md.current_version, ext=src_path.suffix
            )
            futures[url] = executor.submit(
                ObjectOps.put_file_resolved, store, src_path, file_dest, TempStore.contains_path(src_path)
            )
        return {url: future.result() for url, future in futures.items()}


def delete_media(md: TableVersionMd) -> None:
    """Delete the media files of the table with metadata md from all of its destinations."""
    tbl_id = uuid.UUID(md.tbl_md.tbl_id)
    for destination in TableVersion.media_destinations(_store_columns(md)):
        ObjectOps.delete(destination, tbl_id)


def _local_media_dirs(tv: TableVersion, cols: list[Column]) -> list[FsPath]:
    """Return the resolved local directories that contain the stored media files of tv."""
    addrs = [
        ObjectPath.parse_object_storage_addr(dest or str(Env.get().media_dir), False)
        for dest in TableVersion.media_destinations(cols)
    ]
    return [
        (addr.to_path / ObjectPath.table_prefix(tv.id)).resolve()
        for addr in addrs
        if addr.storage_target == StorageTarget.LOCAL_STORE
    ]


def _check_stored_url(tv: TableVersion, url: str, local_media_dirs: list[FsPath]) -> str:
    # A query returns the content of the local file in a cell. Without this check, a row could refer to any file on
    # this machine, such as a credential file. A copy within one database also passes this check, because the export
    # sends every local file of the rows, and the copy replaces each local url with a stored url.
    path = parse_local_file_path(url)
    if path is not None and not any(d in path.resolve().parents for d in local_media_dirs):
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{url!r} is not a media file of {tv.name!r}')
    return url


def insert_rows(tv: TableVersion, md: TableVersionMd, rows_file: str, media_urls: dict[str, str]) -> None:
    """Insert the store rows in rows_file into the store table of tv as rows of the current version of tv. md is the
    metadata of tv. Replace each media url that is a key of media_urls with its value."""
    store = tv.store_tbl
    sa_tbl = store.sa_tbl
    file_cols = pq.ParquetFile(rows_file).schema_arrow.names
    # The rows file has no v_min and v_max. The insert sets v_min, and the default of v_max marks a row of the current
    # version.
    version_col_names = ['v_min', 'v_max'] if tv.is_data_versioned else []
    data_col_names = [col.name for col in sa_tbl.columns if col.name not in version_col_names]
    if set(file_cols) != set(data_col_names):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_DATA_FORMAT,
            f'The rows do not match the columns of {tv.name!r}: {sorted(file_cols)} != {sorted(data_col_names)}',
        )
    insert_cols = [*file_cols, 'v_min'] if tv.is_data_versioned else file_cols
    cols = _store_columns(md)
    media_cols = {col.store_name() for col in cols if col.col_type.is_media_type()}
    cellmd_cols = {col.cellmd_store_name() for col in cols if col.stores_cellmd}
    local_media_dirs = _local_media_dirs(tv, cols)

    for rows in _iter_rows(rows_file):
        store_rows: list[list[Any]] = []
        for row in rows:
            # a rowid that tv has not assigned yet is refused, so that a later insert does not reuse it
            if tv.is_data_versioned and row['rowid'] >= tv.next_row_id:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'Row {row["rowid"]} is not a row of {tv.name!r}'
                )
            store_row: list[Any] = []
            for name in file_cols:
                val = row[name]
                store_val = arrow.arrow_val_to_sa_val(val, sa_tbl.columns[name].type)
                if name in media_cols and val is not None:
                    store_val = _check_stored_url(tv, media_urls.get(store_val, store_val), local_media_dirs)
                elif name in cellmd_cols and val is not None and 'file_urls' in store_val:
                    store_val['file_urls'] = [
                        _check_stored_url(tv, media_urls.get(url, url), local_media_dirs)
                        for url in store_val['file_urls']
                    ]
                store_row.append(store_val)
            if tv.is_data_versioned:
                store_row.append(tv.version)
            store_rows.append(store_row)
        store.sql_insert(sa_tbl, insert_cols, store_rows)

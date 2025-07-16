import base64
import datetime
import io
import itertools
import json
import logging
import tarfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional
from uuid import UUID

import more_itertools
import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, metadata, type_system as ts
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.utils import sha256sum
from pixeltable.utils.formatter import Formatter
from pixeltable.utils.media_store import MediaStore

_logger = logging.getLogger('pixeltable')


class TablePackager:
    """
    Packages a pixeltable Table into a tarball containing Parquet tables and media files. The structure of the tarball
    is as follows:

    metadata.json  # Pixeltable metadata for the packaged table and its ancestors
    tables/**  # Parquet tables for the packaged table and its ancestors, each table in a directory 'tbl_{tbl_id.hex}'
    media/**  # Local media files

    If the table contains media columns, they are handled as follows:
    - If a media file has an external URL (any URL scheme other than file://), then the URL will be preserved as-is and
      stored in the Parquet table.
    - If a media file is a local file, then it will be copied into the tarball as a file of the form
      'media/{uuid}{extension}', and the Parquet table will contain the ephemeral URI 'pxtmedia://{uuid}{extension}'.
    """

    table: catalog.Table  # The table to be packaged
    tmp_dir: Path  # Temporary directory where the package will reside
    tables_dir: Path  # Directory where the Parquet tables will be written
    media_files: dict[Path, str]  # Mapping from local media file paths to their tarball names
    md: dict[str, Any]

    bundle_path: Path
    preview_header: dict[str, str]
    preview: list[list[Any]]

    def __init__(self, table: catalog.Table, additional_md: Optional[dict[str, Any]] = None) -> None:
        self.table = table
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

        # Load metadata
        with catalog.Catalog.get().begin_xact(for_write=False):
            tbl_md = catalog.Catalog.get().load_replica_md(table)
            self.md = {
                'pxt_version': pxt.__version__,
                'pxt_md_version': metadata.VERSION,
                'md': {'tables': [md.as_dict() for md in tbl_md]},
            }
        if additional_md is not None:
            self.md.update(additional_md)

    def package(self) -> Path:
        """
        Export the table to a tarball containing Parquet tables and media files.
        """
        assert not self.tmp_dir.exists()  # Packaging can only be done once per TablePackager instance

        _logger.info(f'Packaging table {self.table._path()!r} and its ancestors in: {self.tmp_dir}')
        self.tmp_dir.mkdir()
        with open(self.tmp_dir / 'metadata.json', 'w', encoding='utf8') as fp:
            json.dump(self.md, fp)
        self.tables_dir = self.tmp_dir / 'tables'
        self.tables_dir.mkdir()
        with catalog.Catalog.get().begin_xact(for_write=False):
            for tv in self.table._tbl_version_path.get_tbl_versions():
                _logger.info(f'Exporting table {tv.get().versioned_name!r}.')
                self.__export_table(tv.get())

        _logger.info('Building archive.')
        self.bundle_path = self.__build_tarball()

        _logger.info('Extracting preview data.')
        self.md['count'] = self.table.count()
        preview_header, preview = self.__extract_preview_data()
        self.md['preview_header'] = preview_header
        self.md['preview'] = preview

        _logger.info(f'Packaging complete: {self.bundle_path}')
        return self.bundle_path

    def __export_table(self, tv: catalog.TableVersion) -> None:
        """
        Exports the data from `t` into a Parquet table.
        """
        # `tv` must be an ancestor of the primary table
        assert any(tv.id == base.id for base in self.table._tbl_version_path.get_tbl_versions())
        sql_types = {col.name: col.type for col in tv.store_tbl.sa_tbl.columns}
        media_cols: set[str] = set()
        for col in tv.cols:
            if col.is_stored and col.col_type.is_media_type():
                media_cols.add(col.store_name())

        parquet_schema = self.__to_parquet_schema(tv.store_tbl.sa_tbl)
        # TODO: Partition larger tables into multiple parquet files. (The parquet file naming scheme anticipates
        #     future support for this.)
        parquet_dir = self.tables_dir / f'tbl_{tv.id.hex}'
        parquet_dir.mkdir()
        parquet_file = parquet_dir / f'tbl_{tv.id.hex}.00000.parquet'
        _logger.info(f'Creating parquet table: {parquet_file}')

        # Populate the Parquet table with data.
        # The data is first loaded from the DataFrame into a sequence of pyarrow tables, batched in order to avoid
        # excessive memory usage. The pyarrow tables are then amalgamated into the (single) Parquet table on disk.
        # We use snappy compression for the Parquet tables; the entire bundle will be bzip2-compressed later, so
        # faster compression should provide good performance while still reducing temporary storage utilization.
        parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='SNAPPY')
        filter_tv = self.table._tbl_version_path.tbl_version.get()
        row_iter = tv.store_tbl.dump_rows(tv.version, filter_tv.store_tbl, filter_tv.version)
        for pa_table in self.__to_pa_tables(row_iter, sql_types, media_cols, parquet_schema):
            parquet_writer.write_table(pa_table)
        parquet_writer.close()

    # The following methods are responsible for schema and data conversion from Pixeltable to Parquet.

    @classmethod
    def __to_parquet_schema(cls, store_tbl: sql.Table) -> pa.Schema:
        entries = [(col_name, cls.__to_parquet_type(col.type)) for col_name, col in store_tbl.columns.items()]
        return pa.schema(entries)  # type: ignore[arg-type]

    @classmethod
    def __to_parquet_type(cls, col_type: sql.types.TypeEngine[Any]) -> pa.DataType:
        if isinstance(col_type, sql.String):
            return pa.string()
        if isinstance(col_type, sql.Boolean):
            return pa.bool_()
        if isinstance(col_type, sql.BigInteger):
            return pa.int64()
        if isinstance(col_type, sql.Float):
            return pa.float32()
        if isinstance(col_type, sql.TIMESTAMP):
            return pa.timestamp('us', tz=datetime.timezone.utc)
        if isinstance(col_type, sql.Date):
            return pa.date32()
        if isinstance(col_type, sql.JSON):
            return pa.string()  # JSON will be exported as strings
        if isinstance(col_type, sql.LargeBinary):
            return pa.binary()
        raise AssertionError(f'Unrecognized SQL type: {col_type} (type {type(col_type)})')

    def __to_pa_tables(
        self,
        row_iter: Iterator[dict[str, Any]],
        sql_types: dict[str, sql.types.TypeEngine[Any]],
        media_cols: set[str],
        arrow_schema: pa.Schema,
        batch_size: int = 1_000,
    ) -> Iterator[pa.Table]:
        """
        Group rows into a sequence of pyarrow tables, batched into smaller chunks to minimize memory utilization.
        The row dictionaries have the format {store_col_name: value}, where the values reflect the unprocessed contents
        of the store database (as returned by `StoreTable.dump_rows()`).
        """
        for rows in more_itertools.batched(row_iter, batch_size):
            cols = {}
            for name, sql_type in sql_types.items():
                is_media_col = name in media_cols
                values = [self.__to_pa_value(row.get(name), sql_type, is_media_col) for row in rows]
                cols[name] = values
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_value(self, val: Any, sql_type: sql.types.TypeEngine[Any], is_media_col: bool) -> Any:
        if val is None:
            return None
        if isinstance(sql_type, sql.JSON):
            # Export JSON as strings
            return json.dumps(val)
        if is_media_col:
            # Handle media files as described above
            assert isinstance(val, str)
            return self.__process_media_url(val)
        return val

    def __process_media_url(self, url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme == 'file':
            # It's the URL of a local file. Replace it with a pxtmedia:// URI.
            # (We can't use an actual pxt:// URI, because the eventual pxt:// table name might not be known at this
            # time. The pxtmedia:// URI serves as a relative reference into the tarball that can be replaced with an
            # actual URL when the table is reconstituted.)
            path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_url.path)))
            if path not in self.media_files:
                # Create a new entry in the `media_files` dict so that we can copy the file into the tarball later.
                # We name the media files in the archive by their SHA256 hash. This ensures that we can properly
                # deduplicate and validate them later.
                # If we get a collision, it's not a problem; it just means we have two identical files (which will
                # be conveniently deduplicated in the bundle).
                sha = sha256sum(path)
                dest_name = f'{sha}{path.suffix}'
                self.media_files[path] = dest_name
            return f'pxtmedia://{self.media_files[path]}'
        # For any type of URL other than a local file, just return the URL as-is.
        return url

    def __build_tarball(self) -> Path:
        bundle_path = self.tmp_dir / 'bundle.tar.bz2'
        with tarfile.open(bundle_path, 'w:bz2') as tf:
            # Add metadata json
            tf.add(self.tmp_dir / 'metadata.json', arcname='metadata.json')
            # Add the dir containing Parquet tables
            tf.add(self.tables_dir, arcname='tables')
            # Add the media files
            for src_file, dest_name in self.media_files.items():
                tf.add(src_file, arcname=f'media/{dest_name}')
        return bundle_path

    def __extract_preview_data(self) -> tuple[dict[str, str], list[list[Any]]]:
        """
        Extract a preview of the table data for display in the UI.

        In order to bound the size of the output data, all "unbounded" data types are resized:
        - Strings are abbreviated as per Formatter.abbreviate()
        - Arrays and JSON are shortened and formatted as strings
        - Images are resized to thumbnail size as a base64-encoded webp
        - Videos are replaced by their first frame and resized as above
        - Documents are replaced by a thumbnail as a base64-encoded webp
        """
        # First 8 columns
        preview_cols = dict(itertools.islice(self.table._get_schema().items(), 0, 8))
        select_list = [self.table[col_name] for col_name in preview_cols]
        # First 5 rows
        rows = list(self.table.select(*select_list).head(n=5))

        preview_header = {col_name: str(col_type._type) for col_name, col_type in preview_cols.items()}
        preview = [
            [self.__encode_preview_data(val, col_type)]
            for row in rows
            for val, col_type in zip(row.values(), preview_cols.values())
        ]

        return preview_header, preview

    def __encode_preview_data(self, val: Any, col_type: ts.ColumnType) -> Any:
        if val is None:
            return None

        match col_type._type:
            case ts.ColumnType.Type.STRING:
                assert isinstance(val, str)
                return Formatter.abbreviate(val)

            case ts.ColumnType.Type.INT | ts.ColumnType.Type.FLOAT | ts.ColumnType.Type.BOOL:
                return val

            case ts.ColumnType.Type.TIMESTAMP | ts.ColumnType.Type.DATE:
                return str(val)

            case ts.ColumnType.Type.ARRAY:
                assert isinstance(val, np.ndarray)
                return Formatter.format_array(val)

            case ts.ColumnType.Type.JSON:
                # We need to escape the JSON string server-side for security reasons.
                # Therefore we don't escape it here, in order to avoid double-escaping.
                return Formatter.format_json(val, escape_strings=False)

            case ts.ColumnType.Type.IMAGE:
                # Rescale the image to minimize data transfer size
                assert isinstance(val, PIL.Image.Image)
                return self.__encode_image(val)

            case ts.ColumnType.Type.VIDEO:
                assert isinstance(val, str)
                return self.__encode_video(val)

            case ts.ColumnType.Type.AUDIO:
                return None

            case ts.ColumnType.Type.DOCUMENT:
                assert isinstance(val, str)
                return self.__encode_document(val)

            case _:
                raise AssertionError(f'Unrecognized column type: {col_type._type}')

    def __encode_image(self, img: PIL.Image.Image) -> str:
        # Heuristic for thumbnail sizing:
        # Standardize on a width of 240 pixels (to most efficiently utilize the columnar display).
        # But, if the aspect ratio is below 2:3, bound the height at 360 pixels (to avoid unboundedly tall thumbnails
        #     in the case of highly oblong images).
        if img.height > img.width * 1.5:
            scaled_img = img.resize((img.width * 360 // img.height, 360))
        else:
            scaled_img = img.resize((240, img.height * 240 // img.width))
        with io.BytesIO() as buffer:
            scaled_img.save(buffer, 'webp')
            return base64.b64encode(buffer.getvalue()).decode()

    def __encode_video(self, video_path: str) -> Optional[str]:
        thumb = Formatter.extract_first_video_frame(video_path)
        return self.__encode_image(thumb) if thumb is not None else None

    def __encode_document(self, doc_path: str) -> Optional[str]:
        thumb = Formatter.make_document_thumbnail(doc_path)
        return self.__encode_image(thumb) if thumb is not None else None


class TableRestorer:
    """
    Creates a replica table from a tarball containing Parquet tables and media files. See the `TablePackager` docs for
    details on the tarball structure.

    Args:
        tbl_path: Pixeltable path (such as 'my_dir.my_table') where the materialized table will be made visible.
        md: Optional metadata dictionary. If not provided, metadata will be read from the tarball's `metadata.json`.
            The metadata contains table_md, table_version_md, and table_schema_version_md entries for each ancestor
            of the table being restored, as written out by `TablePackager`.
    """

    tbl_path: str
    md: Optional[dict[str, Any]]
    tmp_dir: Path
    media_files: dict[str, str]  # Mapping from pxtmedia:// URLs to local file:// URLs

    def __init__(self, tbl_path: str, md: Optional[dict[str, Any]] = None) -> None:
        self.tbl_path = tbl_path
        self.md = md
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

    def restore(self, bundle_path: Path) -> pxt.Table:
        # Extract tarball
        print(f'Extracting table data into: {self.tmp_dir}')
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(path=self.tmp_dir)

        if self.md is None:
            # No metadata supplied; read it from the archive
            with open(self.tmp_dir / 'metadata.json', 'r', encoding='utf8') as fp:
                self.md = json.load(fp)

        pxt_md_version = self.md['pxt_md_version']
        assert isinstance(pxt_md_version, int)

        if pxt_md_version != metadata.VERSION:
            raise excs.Error(
                f'Pixeltable metadata version mismatch: {pxt_md_version} != {metadata.VERSION}.\n'
                'Please upgrade Pixeltable to use this dataset: pip install -U pixeltable'
            )

        tbl_md = [schema.FullTableMd.from_dict(t) for t in self.md['md']['tables']]
        for md in tbl_md:
            md.tbl_md.is_replica = True

        cat = catalog.Catalog.get()

        with cat.begin_xact(for_write=True):
            # Create (or update) the replica table and its ancestors, along with TableVersion instances for any
            # versions that have not been seen before.
            cat.create_replica(catalog.Path(self.tbl_path), tbl_md)

            # Now we need to load data for replica_tbl and its ancestors, except that we skip
            # replica_tbl itself if it's a pure snapshot.
            for md in tbl_md[::-1]:  # Base table first
                if not md.is_pure_snapshot:
                    tv = cat.get_tbl_version(UUID(md.tbl_md.tbl_id), md.version_md.version)
                    # Import data from Parquet.
                    _logger.info(f'Importing table {tv.name!r}.')
                    self.__import_table(self.tmp_dir, tv, md)

            return cat.get_table_by_id(UUID(tbl_md[0].tbl_md.tbl_id))

    def __import_table(self, bundle_path: Path, tv: catalog.TableVersion, tbl_md: schema.FullTableMd) -> None:
        """
        Import the Parquet table into the Pixeltable catalog.
        """
        tbl_id = UUID(tbl_md.tbl_md.tbl_id)
        parquet_dir = bundle_path / 'tables' / f'tbl_{tbl_id.hex}'
        parquet_table = pq.read_table(str(parquet_dir))
        replica_version = tv.version

        conn = Env.get().conn
        store_sa_tbl = tv.store_tbl.sa_tbl
        store_sa_tbl_name = tv.store_tbl._storage_name()

        # Sometimes we are importing a table that has never been seen before. Other times, however, we are importing
        # an existing replica table, and the table version and/or row selection differs from what was imported
        # previously. Care must be taken to ensure that the new data is merged with existing data in a way that
        # yields an internally consistent version history for each row.

        # The overall strategy is this:
        # 1. Import the parquet data into a temporary table;
        # 2. "rectify" the v_max values in both the temporary table and the existing table (more on this below);
        # 3. Delete any row instances from the temporary table that are already present in the existing table;
        # 4. Copy the remaining rows from the temporary table into the existing table.

        # Create a temporary table for the initial data load, containing columns for all columns present in the
        # parquet table. The parquet columns have identical names to those in the store table, so we can use the
        # store table schema to get their SQL types (which are not necessarily derivable from their Parquet types,
        # e.g., pa.string() may hold either VARCHAR or serialized JSONB).
        temp_cols: dict[str, sql.Column] = {}
        for field in parquet_table.schema:
            assert field.name in store_sa_tbl.columns
            col_type = store_sa_tbl.columns[field.name].type
            temp_cols[field.name] = sql.Column(field.name, col_type)
        temp_sa_tbl_name = f'temp_{uuid.uuid4().hex}'
        _logger.debug(f'Creating temporary table: {temp_sa_tbl_name}')
        temp_md = sql.MetaData()
        temp_sa_tbl = sql.Table(temp_sa_tbl_name, temp_md, *temp_cols.values(), prefixes=['TEMPORARY'])
        temp_sa_tbl.create(conn)

        # Populate the temporary table with data from the Parquet file.
        _logger.debug(f'Loading {parquet_table.num_rows} row(s) into temporary table: {temp_sa_tbl_name}')
        for batch in parquet_table.to_batches(max_chunksize=10_000):
            pydict = batch.to_pydict()
            rows = self.__from_pa_pydict(tv, pydict)
            conn.execute(sql.insert(temp_sa_tbl), rows)

        # Each row version is identified uniquely by its pk, a tuple (row_id, pos_0, pos_1, ..., pos_k, v_min).
        # Conversely, v_max is not part of the primary key, but is simply a bookkeeping device.
        # In an original table, v_max is always equal to the v_min of the succeeding row instance with the same
        # row id, or MAX_VERSION if no such row instance exists. But in the replica, we need to be careful, since
        # we might see only a subset of the original table's versions, and we might see them out of order.

        # We'll adjust the v_max values according to the principle of "latest provable v_max":
        # they will always correspond to the latest version for which we can prove the row instance was alive. This
        # will enable us to maintain consistency of the v_max values if additional table versions are later imported,
        # regardless of the order in which they are seen. It also means that replica tables (unlike original tables)
        # may have gaps in their row version histories, but this is fine; the gaps simply correspond to table versions
        # that have never been observed.

        pk_predicates = [col == temp_cols[col.name] for col in tv.store_tbl.pk_columns()]
        pk_clause = sql.and_(*pk_predicates)

        # If the same pk exists in both the temporary table and the existing table, then the corresponding row data
        # must be identical; the rows can differ only in their v_max value. As a sanity check, we go through the
        # motion of verifying this; a failure implies data corruption in either the replica being imported or in a
        # previously imported replica.

        system_col_names = {col.name for col in tv.store_tbl.system_columns()}
        media_col_names = {col.store_name() for col in tv.cols if col.col_type.is_media_type() and col.is_stored}
        value_store_cols = [
            store_sa_tbl.c[col_name]
            for col_name in temp_cols
            if col_name not in system_col_names and col_name not in media_col_names
        ]
        value_temp_cols = [
            col
            for col_name, col in temp_cols.items()
            if col_name not in system_col_names and col_name not in media_col_names
        ]
        mismatch_predicates = [store_col != temp_col for store_col, temp_col in zip(value_store_cols, value_temp_cols)]
        mismatch_clause = sql.or_(*mismatch_predicates)

        # This query looks for rows that have matching primary keys (rowid + pos_k + v_min), but differ in at least
        # one value column. Pseudo-SQL:
        #
        # SELECT store_tbl.col_0, ..., store_tbl.col_n, temp_tbl.col_0, ...,  temp_tbl.col_n
        # FROM store_tbl, temp_tbl
        # WHERE store_tbl.rowid = temp_tbl.rowid
        #     AND store_tbl.pos_0 = temp_tbl.pos_0
        #     AND ... AND store_tbl.pos_k = temp_tbl.pos_k
        #     AND store_tbl.v_min = temp_tbl.v_min
        #     AND (
        #         store_tbl.col_0 != temp_tbl.col_0
        #         OR store_tbl.col_1 != temp_tbl.col_1
        #         OR ... OR store_tbl.col_n != temp_tbl.col_n
        #     )
        #
        # The value column comparisons (store_tbl.col_0 != temp_tbl.col_0, etc.) will always be false for rows where
        # either column is NULL; this is what we want, since it may indicate a column that is present in one version
        # but not the other.
        q = sql.select(*value_store_cols, *value_temp_cols).where(pk_clause).where(mismatch_clause)
        _logger.debug(q.compile())
        result = conn.execute(q)
        if result.rowcount > 0:
            _logger.debug(
                f'Data corruption error between {temp_sa_tbl_name!r} and {store_sa_tbl_name!r}: '
                f'{result.rowcount} inconsistent row(s).'
            )
            row = result.first()
            _logger.debug('Example mismatch:')
            _logger.debug(f'{store_sa_tbl_name}: {row[: len(value_store_cols)]}')
            _logger.debug(f'{temp_sa_tbl_name}: {row[len(value_store_cols) :]}')
            raise excs.Error(
                'Data corruption error: the replica data are inconsistent with data retrieved from a previous replica.'
            )
        _logger.debug(f'Verified data integrity between {store_sa_tbl_name!r} and {temp_sa_tbl_name!r}.')

        # Now rectify the v_max values in the temporary table.
        # If a row instance has a concrete v_max value, then we know it's genuine: it's the unique and immutable
        # version when the row was deleted. (This can only happen if later versions of the base table already
        # existed at the time this replica was published.)
        # But if a row instance has a v_max value of MAX_VERSION, then we don't know anything about its future.
        # It might live indefinitely, or it might be deleted as early as version `n + 1`. Following the principle
        # of "latest provable v_max", we simply set v_max equal to `n + 1`.
        q = (
            temp_sa_tbl.update()
            .values(v_max=(replica_version + 1))
            .where(temp_sa_tbl.c.v_max == schema.Table.MAX_VERSION)
        )
        _logger.debug(q.compile())
        result = conn.execute(q)
        _logger.debug(f'Rectified {result.rowcount} row(s) in {temp_sa_tbl_name!r}.')

        # Now rectify the v_max values in the existing table. This is done by simply taking the later of the two v_max
        # values (the existing one and the new one) for each row instance, following the "latest provable v_max"
        # principle. Obviously we only need to do this for rows that exist in both tables (it's a simple join).
        q = (
            store_sa_tbl.update()
            .values(v_max=sql.func.greatest(store_sa_tbl.c.v_max, temp_sa_tbl.c.v_max))
            .where(pk_clause)
        )
        _logger.debug(q.compile())
        result = conn.execute(q)
        _logger.debug(f'Rectified {result.rowcount} row(s) in {store_sa_tbl_name!r}.')

        # Now we need to update rows in the existing table that are also present in the temporary table. This is to
        # account for the scenario where the temporary table has columns that are not present in the existing table.
        # (We can't simply replace the rows with their versions in the temporary table, because the converse scenario
        # might also occur; there may be columns in the existing table that are not present in the temporary table.)
        value_update_clauses: dict[str, sql.ColumnElement] = {}
        for temp_col in temp_cols.values():
            if temp_col.name not in system_col_names:
                store_col = store_sa_tbl.c[temp_col.name]
                # Prefer the value from the existing table, substituting the value from the temporary table if it's
                # NULL. This works in all cases (including media columns, where we prefer the existing media file).
                clause = sql.case((store_col == None, temp_col), else_=store_col)
                value_update_clauses[temp_col.name] = clause
        if len(value_update_clauses) > 0:
            q = store_sa_tbl.update().values(**value_update_clauses).where(pk_clause)
            _logger.debug(q.compile())
            result = conn.execute(q)
            _logger.debug(
                f'Merged values from {temp_sa_tbl_name!r} into {store_sa_tbl_name!r} for {result.rowcount} row(s).'
            )

        # Now drop any rows from the temporary table that are also present in the existing table.
        # The v_max values have been rectified, data has been merged into NULL cells, and all other row values have
        # been verified identical.
        # TODO: Delete any media files that were orphaned by this operation (they're necessarily duplicates of media
        #     files that are already present in the existing table).
        q = temp_sa_tbl.delete().where(pk_clause)
        _logger.debug(q.compile())
        result = conn.execute(q)
        _logger.debug(f'Deleted {result.rowcount} row(s) from {temp_sa_tbl_name!r}.')

        # Finally, copy the remaining data (consisting entirely of new row instances) from the temporary table into
        # the actual table.
        q = store_sa_tbl.insert().from_select(
            [store_sa_tbl.c[col_name] for col_name in temp_cols], sql.select(*temp_cols.values())
        )
        _logger.debug(q.compile())
        result = conn.execute(q)
        _logger.debug(f'Inserted {result.rowcount} row(s) from {temp_sa_tbl_name!r} into {store_sa_tbl_name!r}.')

    def __from_pa_pydict(self, tv: catalog.TableVersion, pydict: dict[str, Any]) -> list[dict[str, Any]]:
        # Data conversions from pyarrow to Pixeltable
        sql_types: dict[str, sql.types.TypeEngine[Any]] = {}
        for col_name in pydict:
            assert col_name in tv.store_tbl.sa_tbl.columns
            sql_types[col_name] = tv.store_tbl.sa_tbl.columns[col_name].type
        media_col_ids: dict[str, int] = {}
        for col in tv.cols:
            if col.is_stored and col.col_type.is_media_type():
                media_col_ids[col.store_name()] = col.id

        row_count = len(next(iter(pydict.values())))
        rows: list[dict[str, Any]] = []
        for i in range(row_count):
            row = {
                col_name: self.__from_pa_value(tv, col_vals[i], sql_types[col_name], media_col_ids.get(col_name))
                for col_name, col_vals in pydict.items()
            }
            rows.append(row)

        return rows

    def __from_pa_value(
        self, tv: catalog.TableVersion, val: Any, sql_type: sql.types.TypeEngine[Any], media_col_id: Optional[int]
    ) -> Any:
        if val is None:
            return None
        if isinstance(sql_type, sql.JSON):
            return json.loads(val)
        if media_col_id is not None:
            assert isinstance(val, str)
            return self.__relocate_media_file(tv, media_col_id, val)
        return val

    def __relocate_media_file(self, tv: catalog.TableVersion, media_col_id: int, url: str) -> str:
        # If this is a pxtmedia:// URL, relocate it
        parsed_url = urllib.parse.urlparse(url)
        assert parsed_url.scheme != 'file'  # These should all have been converted to pxtmedia:// URLs
        if parsed_url.scheme == 'pxtmedia':
            if url not in self.media_files:
                # First time seeing this pxtmedia:// URL. Relocate the file to the media store and record the mapping
                # in self.media_files.
                src_path = self.tmp_dir / 'media' / parsed_url.netloc
                # Move the file to the media store and update the URL.
                self.media_files[url] = MediaStore.relocate_local_media_file(src_path, tv.id, media_col_id, tv.version)
            return self.media_files[url]
        # For any type of URL other than a local file, just return the URL as-is.
        return url

import datetime
import json
import logging
import tarfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

import more_itertools
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, metadata
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.utils.media_store import MediaStore
from pixeltable.utils.sql import log_explain

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

    def __init__(self, table: catalog.Table, additional_md: Optional[dict[str, Any]] = None) -> None:
        self.table = table
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

        # Load metadata
        with Env.get().begin_xact():
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
        _logger.info(f"Packaging table '{self.table._path}' and its ancestors in: {self.tmp_dir}")
        self.tmp_dir.mkdir()
        with open(self.tmp_dir / 'metadata.json', 'w', encoding='utf8') as fp:
            json.dump(self.md, fp)
        self.tables_dir = self.tmp_dir / 'tables'
        self.tables_dir.mkdir()
        with Env.get().begin_xact():
            for tv in self.table._tbl_version_path.get_tbl_versions():
                _logger.info(f"Exporting table '{tv.get().name}:{tv.get().version}'.")
                self.__export_table(tv.get())
        _logger.info('Building archive.')
        bundle_path = self.__build_tarball()
        _logger.info(f'Packaging complete: {bundle_path}')
        return bundle_path

    def __export_table(self, tv: catalog.TableVersion) -> None:
        """
        Exports the data from `t` into a Parquet table.
        """
        # `tv` must be an ancestor of the primary table
        assert any(tv.id == base.id for base in self.table._tbl_version_path.get_tbl_versions())
        sql_types = {col.name: col.type for col in tv.store_tbl.sa_tbl.columns}
        media_cols: set[str] = set()
        for col in tv.cols_by_name.values():
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
        filter_tv = self.table._tbl_version.get()
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
                dest_name = f'{uuid.uuid4().hex}{path.suffix}'
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

        # Create the replica table
        # TODO: This needs to be made concurrency-safe.
        replica_tbl = catalog.Catalog.get().create_replica(catalog.Path(self.tbl_path), tbl_md)
        assert replica_tbl._tbl_version.get().is_snapshot

        # Now we need to instantiate and load data for replica_tbl and its ancestors, except that we skip
        # replica_tbl itself if it's a pure snapshot.
        if replica_tbl._id != replica_tbl._tbl_version.id:
            ancestor_md = tbl_md[1:]  # Pure snapshot; skip replica_tbl
        else:
            ancestor_md = tbl_md  # Not a pure snapshot; include replica_tbl

        # Instantiate data from the Parquet tables.
        with Env.get().begin_xact():
            for md in ancestor_md[::-1]:  # Base table first
                # Create a TableVersion instance (and a store table) for this ancestor.
                tv = catalog.TableVersion.create_replica(md)
                # Now import data from Parquet.
                _logger.info(f'Importing table {tv.name!r}.')
                self.__import_table(self.tmp_dir, tv, md)

        return replica_tbl

    def __import_table(self, bundle_path: Path, tv: catalog.TableVersion, tbl_md: schema.FullTableMd) -> None:
        """
        Import the Parquet table into the Pixeltable catalog.
        """
        tbl_id = uuid.UUID(tbl_md.tbl_md.tbl_id)
        parquet_dir = bundle_path / 'tables' / f'tbl_{tbl_id.hex}'
        parquet_table = pq.read_table(str(parquet_dir))

        conn = Env.get().conn

        # Create a temporary table to load the data into.
        temp_cols: dict[str, sql.Column] = {}
        for field in parquet_table.schema:
            assert field.name in tv.store_tbl.sa_tbl.columns
            col_type = tv.store_tbl.sa_tbl.columns[field.name].type
            temp_cols[field.name] = sql.Column(field.name, col_type)
        temp_sa_tbl_name = f'temp_{uuid.uuid4().hex}'
        _logger.debug(f'Creating temporary table: {temp_sa_tbl_name}')
        temp_md = sql.MetaData()
        temp_sa_tbl = sql.Table(temp_sa_tbl_name, temp_md, *temp_cols.values())
        temp_sa_tbl.create(conn)

        # Populate the temporary table with data from the Parquet file.
        _logger.debug(f'Loading {parquet_table.num_rows} rows into temporary table: {temp_sa_tbl_name}')
        for batch in parquet_table.to_batches(max_chunksize=10_000):
            pydict = batch.to_pydict()
            rows = self.__from_pa_pydict(tv, pydict)
            conn.execute(sql.insert(temp_sa_tbl), rows)

        # Each row version is identified uniquely by the tuple (row_id, pos_0, pos_1, ..., pos_k, v_min). Conversely,
        # v_max (unlike row_id, pos_i, and v_min) is not part of the primary key, but is simply a bookkeeping device;
        # it must always be equal to the v_min of the succeeding row version for that (row_id, pos_i) tuple. Since not
        # all versions from the original data source are necessarily present in the replica table, we need to "rectify"
        # both the old and new tables by adjusting their v_max values to be internally consistent.

        pk_predicates = [
            col == temp_cols[col.name]
            for col in tv.store_tbl.pk_columns()
        ]
        pk_clause = sql.and_(*pk_predicates)
        rowid_clause = sql.and_(*pk_predicates[:-1])
        vmin_clause = pk_predicates[-1]

        # First look for exact primary key collisions. In such cases, the data between old and new tables must be
        # identical. We double-check this as a failsafe.
        # q = (
        #     sql.select(*temp_cols, *tv.store_tbl.sa_tbl.columns.values())
        #     .join_from(temp_sa_tbl, tv.store_tbl.sa_tbl)
        #     .where(pk_clause)
        # )
        # for row in conn.execute(q).all():

        # Now drop any rows from the temporary table that are exact matches for rows in the actual table.
        q = (
            sql.delete(temp_sa_tbl)
            .where(pk_clause)
        )
        _logger.debug(q.compile())
        result = conn.execute(q)
        _logger.debug(f'Deleted {result.rowcount} rows from {temp_sa_tbl_name!r} that were exact pk matches.')

        # Next, rectify the v_max values in the temporary table.

        # Likewise, rectify the v_max values in the actual table.

        # Finally, copy the data from the temporary table into the actual table, and drop the temporary table.
        sql_text = (
            f'INSERT INTO {tv.store_tbl._storage_name()} ({", ".join(temp_cols.keys())}) '
            f'SELECT {", ".join(temp_cols.keys())} FROM {temp_sa_tbl_name}'
        )
        _logger.debug(sql_text)
        result = conn.execute(sql.text(sql_text))
        _logger.debug(f'Inserted {result.rowcount} rows from {temp_sa_tbl_name!r} into {tv.store_tbl._storage_name()!r}.')

    def __from_pa_pydict(self, tv: catalog.TableVersion, pydict: dict[str, Any]) -> list[dict[str, Any]]:
        # Data conversions from pyarrow to Pixeltable
        sql_types: dict[str, sql.types.TypeEngine[Any]] = {}
        for col_name in pydict:
            assert col_name in tv.store_tbl.sa_tbl.columns
            sql_types[col_name] = tv.store_tbl.sa_tbl.columns[col_name].type
        media_col_ids: dict[str, int] = {}
        for col in tv.cols_by_name.values():
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
                dest_path = MediaStore.prepare_media_path(tv.id, media_col_id, tv.version, ext=src_path.suffix)
                src_path.rename(dest_path)
                self.media_files[url] = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
            return self.media_files[url]
        # For any type of URL other than a local file, just return the URL as-is.
        return url

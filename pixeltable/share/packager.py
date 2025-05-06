import io
import json
import logging
import tarfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

import more_itertools
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy as sql

import pixeltable as pxt
from pixeltable import catalog, exceptions as excs, exprs, metadata, type_system as ts
from pixeltable.dataframe import DataFrame
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.utils.arrow import PXT_TO_PA_TYPES
from pixeltable.utils.media_store import MediaStore

_logger = logging.getLogger('pixeltable')


class TablePackager:
    """
    Packages a pixeltable Table into a tarball containing Parquet tables and media files. The structure of the tarball
    is as follows:

    metadata.json  # Pixeltable metadata for the packaged table and its ancestors
    tables/**  # Parquet tables for the packaged table and its ancestors, named as 'tbl_{tbl_id.hex}.parquet'
    media/**  # Local media files

    All rows will be exported with an additional 'pk' column. Currently, only the most recent version of the table can
    be exported, and only the full table contents.

    Columns in the Parquet tables follow the standard naming conventions:
    - val_{col_name} for the values in stored columns
    - errortype_{col_name} and errormsg_{col_name} for the error columns (if `col.records_errors`)
    - pk for the primary key column

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
            for tvp in self.table._tbl_version_path.ancestors:
                _logger.info(f"Exporting table '{tvp.tbl_version.get().name}:{tvp.tbl_version.get().version}'.")
                self.__export_table(tvp.tbl_version.get())
        _logger.info('Building archive.')
        bundle_path = self.__build_tarball()
        _logger.info(f'Packaging complete: {bundle_path}')
        return bundle_path

    def __export_table(self, tv: catalog.TableVersion) -> None:
        """
        Exports the data from `t` into a Parquet table.
        """
        sql_types = {
            col.name: col.type
            for col in tv.store_tbl.sa_tbl.columns
        }
        media_cols: set[str] = set()
        for col in tv.cols_by_name.values():
            if col.col_type.is_media_type():
                media_cols.add(col.store_name())

        parquet_schema = self.__to_parquet_schema(tv.store_tbl.sa_tbl)
        # The parquet file naming scheme anticipates future support for partitioning.
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

    # The following methods are responsible for schema and data conversion from Pixeltable to Parquet. Some of this
    # logic might be consolidated into arrow.py and unified with general Parquet export, but there are several
    # major differences:
    # - We export all arrays as binary blobs
    # - We include a 'pk' column in the Parquet table
    # - errortype / errormsg are exported with special handling
    # - Media columns are handled specially as indicated above

    @classmethod
    def __to_parquet_schema(cls, store_tbl: sql.Table) -> pa.Schema:
        entries = [
            (col_name, cls.__to_parquet_type(col.type))
            for col_name, col in store_tbl.columns.items()
        ]
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
            return pa.timestamp('us')
        if isinstance(col_type, sql.JSON):
            return pa.string()  # JSON will be exported as strings
        if isinstance(col_type, sql.LargeBinary):
            return pa.binary()
        raise AssertionError(f'Unrecognized SQL type: {col_type} (type {type(col_type)})')

    def __to_pa_tables(
        self,
        row_iter: Iterator[tuple[str, Any]],
        sql_types: dict[str, sql.types.TypeEngine[Any]],
        media_cols: set[str],
        arrow_schema: pa.Schema,
        batch_size: int = 1_000,
    ) -> Iterator[pa.Table]:
        """
        Load a DataFrame as a sequence of pyarrow tables. The pyarrow tables are batched into smaller chunks
        to avoid excessive memory usage.
        """
        for rows in more_itertools.batched(row_iter, batch_size):
            cols = {}
            for name, sql_type in sql_types.items():
                is_media_col = name in media_cols
                values = [self.__to_pa_value(row.get(name), sql_type, is_media_col) for row in rows]
                cols[name] = values
            print(list(cols.keys()))
            print([(name, v[0]) for name, v in cols.items()])
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_value(self, val: Any, sql_type: sql.types.TypeEngine[Any], is_media_col: bool) -> Any:
        if val is None:
            return None
        # if col_type.is_array_type():
        #     # Export arrays as binary
        #     assert isinstance(val, np.ndarray)
        #     arr = io.BytesIO()
        #     np.save(arr, val)
        #     return arr.getvalue()
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
            # Read metadata from the archive
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

        for batch in parquet_table.to_batches():
            pydict = batch.to_pydict()
            col_ids: list[Optional[int]] = []
            col_types: list[ts.ColumnType] = []
            for name in pydict:
                if name.startswith('val_'):
                    col = tv.cols_by_name[name.removeprefix('val_')]
                    col_ids.append(col.id)
                    col_types.append(col.col_type)
                elif name.startswith('errortype_') or name.startswith('errormsg_'):
                    col_ids.append(None)
                    col_types.append(ts.StringType())
            rows = self.__from_pa_pydict(tv, pydict, col_ids, col_types)
            tv.store_tbl.insert_replica_rows(rows)

    def __from_pa_pydict(
        self,
        tv: catalog.TableVersion,
        pydict: dict[str, Any],
        col_ids: list[Optional[int]],
        col_types: list[ts.ColumnType],
    ) -> list[dict[str, Any]]:
        # pydict must have length exactly 1 more than col_types, because the pk column is not included in col_types
        assert len(pydict) == len(col_types) + 1, (
            f'{len(pydict)} != {len(col_types) + 1}:\n{list(pydict.keys())}\n{col_types}'
        )
        row_count = len(next(iter(pydict.values())))

        # Data conversions from pyarrow to Pixeltable
        converted_pydict = {
            col_name: [self.__from_pa_value(val, tv, col_id, col_type) for val in col_vals]
            for (col_name, col_vals), col_id, col_type in zip(pydict.items(), col_ids, col_types)
            if col_name != 'pk'
        }
        converted_pydict['pk'] = pydict['pk']  # pk values are kept as lists of integers

        rows = [{col_name: col_vals[i] for col_name, col_vals in converted_pydict.items()} for i in range(row_count)]

        return rows

    def __from_pa_value(self, val: Any, tv: catalog.TableVersion, col_id: int, col_type: ts.ColumnType) -> Any:
        if val is None:
            return None
        if col_type.is_array_type():
            assert isinstance(val, bytes)
            # Decode the value to validate that it represents a valid numpy array ...
            arr = io.BytesIO(val)
            res = np.load(arr)
            assert isinstance(res, np.ndarray)
            # ... but just return the raw bytes, since we'll be direct-inserting them into the db
            return val
        if col_type.is_json_type():
            return json.loads(val)
        if col_type.is_media_type():
            assert isinstance(val, str)
            return self.__relocate_media_file(tv, col_id, val)
        return val

    def __relocate_media_file(self, tv: catalog.TableVersion, col_id: int, url: str) -> str:
        # If this is a pxtmedia:// URL, relocate it
        parsed_url = urllib.parse.urlparse(url)
        assert parsed_url.scheme != 'file'  # These should all have been converted to pxtmedia:// URLs
        if parsed_url.scheme == 'pxtmedia':
            if url not in self.media_files:
                # First time seeing this pxtmedia:// URL. Relocate the file to the media store and record the mapping
                # in self.media_files.
                src_path = self.tmp_dir / 'media' / parsed_url.netloc
                dest_path = MediaStore.prepare_media_path(tv.id, col_id, tv.version, ext=src_path.suffix)
                src_path.rename(dest_path)
                self.media_files[url] = urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest_path)))
            return self.media_files[url]
        # For any type of URL other than a local file, just return the URL as-is.
        return url

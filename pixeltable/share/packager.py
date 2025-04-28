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
                _logger.info(f"Exporting table '{tvp.tbl_version.get().name}'.")
                self.__export_table(tvp)
        _logger.info('Building archive.')
        bundle_path = self.__build_tarball()
        _logger.info(f'Packaging complete: {bundle_path}')
        return bundle_path

    def __export_table(self, tvp: catalog.TableVersionPath) -> None:
        """
        Exports the data from `t` into a Parquet table.
        """
        import pixeltable.functions as pxtf

        # First generate a select list for the data we want to extract from `t`. This includes:
        # - all stored columns, including computed columns;
        # - errortype and errormsg fields whenever they're defined;
        # - primary key columns (rowid components and vmin).
        # We select only those columns that are defined in this table (columns inherited from ancestor tables will be
        # handled separately).
        # For media columns, we substitute `col.fileurl` so that we always get the URL (which may be a file:// URL;
        # these will be specially handled later)
        select_exprs: dict[str, exprs.Expr] = {}

        # As we generate the select list, we construct a separate list of column types. We can't rely on df._schema
        # to get the column types, since we'll be substituting `fileurl`s for media columns.
        actual_col_types: list[ts.ColumnType] = []

        tv = tvp.tbl_version.get()
        for col_name, col in tv.cols_by_name.items():
            if not col.is_stored:
                continue
            col_ref = exprs.ColumnRef(col)
            if col.col_type.is_media_type():
                select_exprs[f'val_{col_name}'] = col_ref.fileurl
            else:
                select_exprs[f'val_{col_name}'] = col_ref
            actual_col_types.append(col.col_type)
            if col.records_errors:
                select_exprs[f'errortype_{col_name}'] = col_ref.errortype
                actual_col_types.append(ts.StringType())
                select_exprs[f'errormsg_{col_name}'] = col_ref.errormsg
                actual_col_types.append(ts.StringType())

        # Add columns for the primary key components of the base table rows.
        # We need to use a VminRef for the vmin component, to ensure we get the correct vmin for the base table
        # (which may be different from the vmin of the primary table).
        # We also explicitly select the rowid components, in order to use them with the group_by() / any_value()
        # pattern to deduplicate in SQL (see below).
        rowid_len = len(tv.store_tbl.rowid_columns())
        for idx in range(rowid_len):
            select_exprs[f'pk_{idx}'] = exprs.RowidRef(tvp.tbl_version, idx)
        select_exprs['pk_vmin'] = exprs.VminRef(tvp.tbl_version)

        if tv.id == self.table._tbl_version.id:
            # Selecting from the primary table: just a simple select statement
            df = self.table.select(**select_exprs)
        else:
            # Selecting from an ancestor table: in this case we still need to run the select() with `self.table` as the
            # context, not the base TableVersionPath, so that we export only those rows that are actually present in
            # `self.table`. The group_by() is needed to handle the case of iterator views correctly; it ensures that
            # records of the base table are appropriately deduplicated on rowid.
            select_exprs = {name: pxtf.any_value(expr) for name, expr in select_exprs.items()}
            df = self.table._df().group_by(tv).select(**select_exprs)

        parquet_schema = self.__to_parquet_schema(df._schema)
        parquet_file = self.tables_dir / f'tbl_{tv.id.hex}.parquet'
        _logger.info(f'Creating parquet table: {parquet_file}')

        # Populate the Parquet table with data.
        # The data is first loaded from the DataFrame into a sequence of pyarrow tables, batched in order to avoid
        # excessive memory usage. The pyarrow tables are then amalgamated into the (single) Parquet table on disk.
        # We use snappy compression for the Parquet tables; the entire bundle will be bzip2-compressed later, so
        # faster compression should provide good performance while still reducing temporary storage utilization.
        parquet_writer = pq.ParquetWriter(parquet_file, parquet_schema, compression='SNAPPY')
        for pa_table in self.__to_pa_tables(df, rowid_len, actual_col_types, parquet_schema):
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
    def __to_parquet_schema(cls, pxt_schema: dict[str, ts.ColumnType]) -> pa.Schema:
        entries = [
            (col_name, cls.__to_parquet_type(col_type))
            for col_name, col_type in pxt_schema.items()
            if not col_name.startswith('pk_')
        ]
        entries.append(('pk', pa.list_(pa.int64())))
        return pa.schema(entries)  # type: ignore[arg-type]

    @classmethod
    def __to_parquet_type(cls, col_type: ts.ColumnType) -> pa.DataType:
        if col_type.is_array_type():
            return pa.binary()
        if col_type.is_media_type():
            return pa.string()
        return PXT_TO_PA_TYPES.get(col_type.__class__)

    def __to_pa_tables(
        self,
        df: DataFrame,
        rowid_len: int,
        actual_col_types: list[ts.ColumnType],
        arrow_schema: pa.Schema,
        batch_size: int = 1_000,
    ) -> Iterator[pa.Table]:
        """
        Load a DataFrame as a sequence of pyarrow tables. The pyarrow tables are batched into smaller chunks
        to avoid excessive memory usage.
        """
        for rows in more_itertools.batched(self.__to_pa_rows(df, rowid_len, actual_col_types), batch_size):
            cols = {
                col_name: [row[idx] for row in rows]
                for idx, col_name in enumerate(df._schema.keys())
                if not col_name.startswith('pk_')
            }
            cols['pk'] = [row[-1] for row in rows]
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_rows(self, df: DataFrame, rowid_len: int, actual_col_types: list[ts.ColumnType]) -> Iterator[list]:
        val_exprs = df._select_list_exprs[: -rowid_len - 1]
        pk_exprs = df._select_list_exprs[-rowid_len - 1 :]
        for row in df._exec():
            vals = [row[e.slot_idx] for e in val_exprs]
            assert len(vals) == len(actual_col_types)
            result = [self.__to_pa_value(val, col_type) for val, col_type in zip(vals, actual_col_types)]
            pk = tuple(row[e.slot_idx] for e in pk_exprs)
            result.append(pk)
            yield result

    def __to_pa_value(self, val: Any, col_type: ts.ColumnType) -> Any:
        if val is None:
            return None
        if col_type.is_array_type():
            # Export arrays as binary
            assert isinstance(val, np.ndarray)
            arr = io.BytesIO()
            np.save(arr, val)
            return arr.getvalue()
        if col_type.is_json_type():
            # Export JSON as strings
            return json.dumps(val)
        if col_type.is_media_type():
            # Handle media files as described above
            assert isinstance(val, str)  # Media columns are always referenced by `fileurl`
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
        parquet_file = bundle_path / 'tables' / f'tbl_{tbl_id.hex}.parquet'
        parquet_table = pq.read_table(str(parquet_file))

        for batch in parquet_table.to_batches():
            pydict = batch.to_pydict()
            col_ids: list[Optional[int]] = []
            col_types: list[pxt.ColumnType] = []
            for name in pydict:
                if name.startswith('val_'):
                    col = tv.cols_by_name[name.removeprefix('val_')]
                    col_ids.append(col.id)
                    col_types.append(col.col_type)
                elif name.startswith('errortype_') or name.startswith('errormsg_'):
                    col_ids.append(None)
                    col_types.append(pxt.StringType())
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

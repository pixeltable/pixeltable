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
import pyiceberg.catalog

import pixeltable as pxt
from pixeltable import catalog, exprs, metadata, type_system as ts
from pixeltable.dataframe import DataFrame
from pixeltable.env import Env
from pixeltable.metadata import schema
from pixeltable.utils.arrow import PXT_TO_PA_TYPES
from pixeltable.utils.iceberg import sqlite_catalog
from pixeltable.utils.media_store import MediaStore

_logger = logging.getLogger('pixeltable')


class TablePackager:
    """
    Packages a pixeltable Table into a tarball containing Iceberg tables and media files. The structure of the tarball
    is as follows:

    metadata.json  # Pixeltable metadata for the packaged table
    warehouse/catalog.db  # sqlite Iceberg catalog
    warehouse/pxt.db/**  # Iceberg metadata and data files (parquet/avro/json)
    media/**  # Local media files

    If the table being archived is a view, then the Iceberg catalog will contain separate tables for the view and each
    of its ancestors. All rows will be exported with additional _rowid and _v_min columns. Currently, only the most
    recent version of the table can be exported, and only the full table contents.

    If the table contains media columns, they are handled as follows:
    - If a media file has an external URL (any URL scheme other than file://), then the URL will be preserved as-is and
      stored in the Iceberg table.
    - If a media file is a local file, then it will be copied into the tarball as a file of the form
      'media/{uuid}{extension}', and the Iceberg table will contain the ephemeral URI 'pxtmedia://{uuid}{extension}'.
    """

    table: catalog.Table  # The table to be packaged
    tmp_dir: Path  # Temporary directory where the package will reside
    iceberg_catalog: pyiceberg.catalog.Catalog
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
        Export the table to a tarball containing Iceberg tables and media files.
        """
        assert not self.tmp_dir.exists()  # Packaging can only be done once per TablePackager instance
        _logger.info(f"Packaging table '{self.table._path()}' and its ancestors in: {self.tmp_dir}")
        self.tmp_dir.mkdir()
        with open(self.tmp_dir / 'metadata.json', 'w', encoding='utf8') as fp:
            json.dump(self.md, fp)
        self.iceberg_catalog = sqlite_catalog(self.tmp_dir / 'warehouse')
        with Env.get().begin_xact():
            ancestors = (self.table, *self.table._bases)
            for t in ancestors:
                _logger.info(f"Exporting table '{t._path()}'.")
                self.__export_table(t)
        _logger.info('Building archive.')
        bundle_path = self.__build_tarball()
        _logger.info(f'Packaging complete: {bundle_path}')
        return bundle_path

    def __export_table(self, t: catalog.Table) -> None:
        """
        Exports the data from `t` into an Iceberg table.
        """
        # First generate a select list for the data we want to extract from `t`. This includes:
        # - all stored columns, including computed columns;
        # - errortype and errormsg fields whenever they're defined.
        # We select only those columns that are defined in this table (columns inherited from ancestor tables will be
        # handled separately).
        # For media columns, we substitute `col.fileurl` so that we always get the URL (which may be a file:// URL;
        # these will be specially handled later)
        select_exprs: dict[str, exprs.Expr] = {}

        # As we generate the select list, we construct a separate list of column types. We can't rely on df._schema
        # to get the column types, since we'll be substituting `fileurl`s for media columns.
        actual_col_types: list[ts.ColumnType] = []

        for col_name, col in t._tbl_version.get().cols_by_name.items():
            if not col.is_stored:
                continue
            if col.col_type.is_media_type():
                select_exprs[f'val_{col_name}'] = t[col_name].fileurl
            else:
                select_exprs[f'val_{col_name}'] = t[col_name]
            actual_col_types.append(col.col_type)
            if col.records_errors:
                select_exprs[f'errortype_{col_name}'] = t[col_name].errortype
                actual_col_types.append(ts.StringType())
                select_exprs[f'errormsg_{col_name}'] = t[col_name].errormsg
                actual_col_types.append(ts.StringType())

        # Run the select() on `self.table`, not `t`, so that we export only those rows that are actually present in
        # `self.table`.
        df = self.table.select(**select_exprs)
        self.iceberg_catalog.create_namespace_if_not_exists('pxt')
        iceberg_schema = self.__to_iceberg_schema(df._schema)
        iceberg_tbl_name = f'pxt.tbl_{t._tbl_version.id.hex}'
        _logger.info(f'Creating iceberg table: {iceberg_tbl_name}')
        iceberg_tbl = self.iceberg_catalog.create_table(iceberg_tbl_name, schema=iceberg_schema)

        # Populate the Iceberg table with data.
        # The data is first loaded from the DataFrame into a sequence of pyarrow tables, batched in order to avoid
        # excessive memory usage. The pyarrow tables are then amalgamated into the (single) Iceberg table on disk.
        for pa_table in self.__to_pa_tables(df, actual_col_types, iceberg_schema):
            iceberg_tbl.append(pa_table)

    # The following methods are responsible for schema and data conversion from Pixeltable to Iceberg. Some of this
    # logic might be consolidated into arrow.py and unified with general Parquet export, but there are several
    # major differences:
    # - Iceberg has no array type; we export all arrays as binary blobs
    # - We include _rowid and _v_min columns in the Iceberg table
    # - Media columns are handled specially as indicated above

    @classmethod
    def __to_iceberg_schema(cls, pxt_schema: dict[str, ts.ColumnType]) -> pa.Schema:
        entries = [(name, cls.__to_iceberg_type(col_type)) for name, col_type in pxt_schema.items()]
        entries.append(('pk', pa.list_(pa.int64())))
        return pa.schema(entries)  # type: ignore[arg-type]

    @classmethod
    def __to_iceberg_type(cls, col_type: ts.ColumnType) -> pa.DataType:
        if col_type.is_array_type():
            return pa.binary()
        if col_type.is_media_type():
            return pa.string()
        return PXT_TO_PA_TYPES.get(col_type.__class__)

    def __to_pa_tables(
        self, df: DataFrame, actual_col_types: list[ts.ColumnType], arrow_schema: pa.Schema, batch_size: int = 1_000
    ) -> Iterator[pa.Table]:
        """
        Load a DataFrame as a sequence of pyarrow tables. The pyarrow tables are batched into smaller chunks
        to avoid excessive memory usage.
        """
        for rows in more_itertools.batched(self.__to_pa_rows(df, actual_col_types), batch_size):
            cols = {col_name: [row[idx] for row in rows] for idx, col_name in enumerate(df._schema.keys())}
            cols['pk'] = [row[-1] for row in rows]
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_rows(self, df: DataFrame, actual_col_types: list[ts.ColumnType]) -> Iterator[list]:
        for row in df._exec():
            vals = [row[e.slot_idx] for e in df._select_list_exprs]
            result = [self.__to_pa_value(val, col_type) for val, col_type in zip(vals, actual_col_types)]
            result.append(row.pk)
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
            # Add the Iceberg warehouse dir (including the catalog)
            tf.add(self.tmp_dir / 'warehouse', arcname='warehouse', recursive=True)
            # Add the media files
            for src_file, dest_name in self.media_files.items():
                tf.add(src_file, arcname=f'media/{dest_name}')
        return bundle_path


class TableRestorer:
    tbl_path: str
    md: Optional[list[schema.FullTableMd]]
    tmp_dir: Path
    media_files: dict[str, Path]  # Mapping from pxtmedia:// URLs to local file paths

    def __init__(self, tbl_path: str, md: Optional[list[schema.FullTableMd]] = None) -> None:
        self.tbl_path = tbl_path
        self.md = md
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

    def restore(self, bundle_path: Path) -> None:
        # Extract tarball
        with tarfile.open(bundle_path, 'r:bz2') as tf:
            tf.extractall(path=self.tmp_dir)

        if self.md is None:
            # Read metadata from the archive
            with open(self.tmp_dir / 'metadata.json', 'r', encoding='utf8') as fp:
                md_json = json.load(fp)
                self.md = [schema.FullTableMd.from_dict(t) for t in md_json['md']['tables']]

        # TODO: Version check

        # Create the replica table
        replica_tbl = catalog.Catalog.get().create_replica(catalog.Path(self.tbl_path), self.md)
        assert replica_tbl._tbl_version.get().is_snapshot

        # Now create TableVersions (and store tables) for the table and all its ancestors.
        with Env.get().begin_xact():
            for md in self.md[::-1]:
                catalog.TableVersion.create_replica(md)

        # Load the Iceberg catalog
        iceberg_catalog = sqlite_catalog(self.tmp_dir / 'warehouse')
        with Env.get().begin_xact():
            ancestors: tuple[catalog.TableVersionHandle, ...] = (
                replica_tbl._tbl_version,
                *replica_tbl._tbl_version_path.get_bases(),
            )
            if replica_tbl._id == replica_tbl._tbl_version.id:
                # replica_tbl is not a pure snapshot. We need to load data for replica_tbl as well as it ancestors.
                ancestor_md = self.md
            else:
                # replica_tbl is a pure snapshot. We'll discard its metadata for purposes of data loading,
                # since the pure snapshot (as a schema object) won't contain any data of its own.
                ancestor_md = self.md[1:]
            assert len(ancestors) == len(ancestor_md)
            for tv, md in zip(ancestors[::-1], ancestor_md[::-1]):
                assert isinstance(tv, catalog.TableVersionHandle)
                _logger.info(f'Importing table {tv.get().name!r}.')
                self.__import_table(iceberg_catalog, tv.get(), md)

    def __import_table(
        self, iceberg_catalog: pyiceberg.catalog.Catalog, tv: catalog.TableVersion, tbl_md: schema.FullTableMd
    ) -> None:
        """
        Import the Iceberg table into the Pixeltable catalog.
        """
        iceberg_tbl_id = uuid.UUID(tbl_md.tbl_md.tbl_id)
        iceberg_tbl_name = f'pxt.tbl_{iceberg_tbl_id.hex}'
        iceberg_tbl = iceberg_catalog.load_table(iceberg_tbl_name)

        for batch in iceberg_tbl.scan().to_arrow_batch_reader():
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
            # Validate that the value represents a valid numpy array
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
            path = self.tmp_dir / 'media' / parsed_url.netloc
            dest = MediaStore.prepare_media_path(tv.id, col_id, tv.version, ext=path.suffix)
            path.rename(dest)
            return urllib.parse.urljoin('file:', urllib.request.pathname2url(str(dest)))
        # For any type of URL other than a local file, just return the URL as-is.
        return url

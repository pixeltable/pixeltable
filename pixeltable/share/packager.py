import io
import json
import tarfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator

import more_itertools
import numpy as np
import pyarrow as pa
from pyiceberg.catalog import Catalog

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.utils.arrow import _pt_to_pa
from pixeltable.utils.iceberg import sqlite_catalog


class TablePackager:
    """
    Packages a pixeltable Table into a tarball containing Iceberg tables and media files. The structure of the tarball
    is as follows:

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

    table: pxt.Table  # The table to be packaged
    tmp_dir: Path  # Temporary directory where the package will reside
    iceberg_catalog: Catalog
    media_files: dict[Path, str]  # Mapping from local media file paths to their tarball names

    def __init__(self, table: pxt.Table) -> None:
        self.table = table
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

    def package(self) -> Path:
        """
        Export the table to a tarball containing Iceberg tables and media files.
        """
        assert not self.tmp_dir.exists()  # Packaging can only be done once per TablePackager instance
        self.tmp_dir.mkdir()
        self.iceberg_catalog = sqlite_catalog(self.tmp_dir / 'warehouse')
        ancestors = [self.table] + self.table._bases
        for t in ancestors:
            self.__export_table(t)
        bundle_path = self.__build_tarball()
        return bundle_path

    def __export_table(self, t: pxt.Table) -> None:
        # Select only those columns that are defined in this table (columns inherited from ancestor
        # tables will be handled separately)
        # TODO: This is selecting only named columns; do we also want to preserve system columns such as errortype?
        col_refs = [
            t[col_name] for col_name, col in t._tbl_version.cols_by_name.items() if not col.col_type.is_media_type()
        ]
        # For media columns, we substitute `col.fileurl` so that we always get the URL (which may be a file:// URL;
        # these will be specially handled later)
        media_col_refs = {
            col_name: t[col_name].fileurl
            for col_name, col in t._tbl_version.cols_by_name.items()
            if col.col_type.is_media_type()
        }
        # Run the select() on `self.table`, not `t`, so that we export only those rows that are actually present in
        # `self.table`.
        df = self.table.select(*col_refs, **media_col_refs)
        namespace = self.__iceberg_namespace(t)
        self.iceberg_catalog.create_namespace_if_not_exists(namespace)
        iceberg_schema = self.__to_iceberg_schema(df._schema)
        iceberg_tbl = self.iceberg_catalog.create_table(f'{namespace}.{t._name}', schema=iceberg_schema)

        # We can't rely on df._schema for the column types, since we substituted `fileurl`s for media columns.
        # Separately construct a list of actual column types in the same order of appearance as in the data frame.
        actual_col_types = [
            col.col_type for col in t._tbl_version.cols_by_name.values() if not col.col_type.is_media_type()
        ]
        actual_col_types.extend(
            col.col_type for col in t._tbl_version.cols_by_name.values() if col.col_type.is_media_type()
        )

        # Populate the Iceberg table with data.
        for pa_table in self.__to_pa_tables(df, actual_col_types, iceberg_schema):
            iceberg_tbl.append(pa_table)

    @classmethod
    def __iceberg_namespace(cls, table: pxt.Table) -> str:
        """
        Iceberg tables must have a namespace, which cannot be the empty string, so we prepend `pxt` to the table path.
        """
        parent_path = table._parent._path
        if len(parent_path) == 0:
            return 'pxt'
        else:
            return f'pxt.{parent_path}'

    # The following methods are responsible for schema and data conversion from Pixeltable to Iceberg. Some of this
    # logic might be consolidated into arrow.py and unified with general Parquet export, but there are several
    # major differences:
    # - Iceberg has no array type; we export all arrays as binary blobs
    # - We include _rowid and _v_min columns in the Iceberg table
    # - Media columns are handled specially as indicated above

    @classmethod
    def __to_iceberg_schema(cls, pxt_schema: dict[str, ts.ColumnType]) -> pa.Schema:
        entries = [(name, cls.__to_iceberg_type(col_type)) for name, col_type in pxt_schema.items()]
        entries.append(('_rowid', pa.list_(pa.int64())))
        entries.append(('_v_min', pa.int64()))
        return pa.schema(entries)  # type: ignore[arg-type]

    @classmethod
    def __to_iceberg_type(cls, col_type: ts.ColumnType) -> pa.DataType:
        if col_type.is_array_type():
            return pa.binary()
        if col_type.is_media_type():
            return pa.string()
        return _pt_to_pa.get(col_type.__class__)

    def __to_pa_tables(
        self,
        df: pxt.DataFrame,
        actual_col_types: list[pxt.ColumnType],
        arrow_schema: pa.Schema,
        batch_size: int = 1_000,
    ) -> Iterator[pa.Table]:
        """
        Export a DataFrame to a sequence of pyarrow tables.
        """
        for rows in more_itertools.batched(self.__to_pa_rows(df, actual_col_types), batch_size):
            cols = {col_name: [row[idx] for row in rows] for idx, col_name in enumerate(df._schema.keys())}
            cols['_rowid'] = [row[-2] for row in rows]
            cols['_v_min'] = [row[-1] for row in rows]
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_rows(self, df: pxt.DataFrame, actual_col_types: list[pxt.ColumnType]) -> Iterator[list]:
        for row in df._exec():
            vals = [row[e.slot_idx] for e in df._select_list_exprs]
            result = [self.__to_pa_value(val, col_type) for val, col_type in zip(vals, actual_col_types)]
            result.append(row.rowid)
            result.append(row.v_min)
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
            # Add the Iceberg warehouse dir (including the catalog)
            tf.add(self.tmp_dir / 'warehouse', arcname='warehouse', recursive=True)
            # Add the media files
            for src_file, dest_name in self.media_files.items():
                tf.add(src_file, arcname=f'media/{dest_name}')
        return bundle_path

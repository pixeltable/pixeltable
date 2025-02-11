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

    table: pxt.Table
    tmp_dir: Path
    iceberg_catalog: Catalog
    media_files: dict[Path, str]

    def __init__(self, table: pxt.Table) -> None:
        self.table = table
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.media_files = {}

    def package(self) -> Path:
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
        col_refs = [t[col_name] for col_name, col in t._tbl_version.cols_by_name.items() if not col.col_type.is_media_type()]
        # For media columns, always use the URL (which may be a file:// URL, but not image data or a cached file
        # reference)
        media_col_refs = {
            col_name: t[col_name].fileurl for col_name, col in t._tbl_version.cols_by_name.items() if col.col_type.is_media_type()
        }
        df = t.select(*col_refs, **media_col_refs)
        namespace = self.__iceberg_namespace(t)
        self.iceberg_catalog.create_namespace_if_not_exists(namespace)
        arrow_schema = self.__to_iceberg_schema(df._schema)
        iceberg_tbl = self.iceberg_catalog.create_table(f'{namespace}.{t._name}', schema=arrow_schema)

        actual_col_types = [col.col_type for col in t._tbl_version.cols_by_name.values() if not col.col_type.is_media_type()]
        actual_col_types.extend(col.col_type for col in t._tbl_version.cols_by_name.values() if col.col_type.is_media_type())
        for pa_table in self.__to_pa_tables(df, actual_col_types, arrow_schema):
            iceberg_tbl.append(pa_table)

    @classmethod
    def __iceberg_namespace(cls, table: pxt.Table) -> str:
        """Iceberg tables must have a namespace, so we prepend `pxt` to the table path."""
        parent_path = table._parent._path
        if len(parent_path) == 0:
            return 'pxt'
        else:
            return f'pxt.{parent_path}'

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

    def __to_pa_tables(self, df: pxt.DataFrame, actual_col_types: list[pxt.ColumnType], arrow_schema: pa.Schema, batch_size: int = 1_000) -> Iterator[pa.Table]:
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
            assert isinstance(val, np.ndarray)
            arr = io.BytesIO()
            np.save(arr, val)
            return arr.getvalue()
        if col_type.is_json_type():
            return json.dumps(val)  # Export JSON as strings
        if col_type.is_media_type():
            assert isinstance(val, str)  # Media columns are always referenced by `fileurl`
            return self.__process_media_url(val)
        return val

    def __process_media_url(self, url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme == 'file':
            # It's the URL of a local file. Make a copy of the file in the archive location and return
            # a pxt:// URI that identifies the copied file.
            path = Path(urllib.parse.unquote(urllib.request.url2pathname(parsed_url.path)))
            if path not in self.media_files:
                dest_name = f'{uuid.uuid4().hex}{path.suffix}'
                self.media_files[path] = dest_name
            return f'pxtmedia://{self.media_files[path]}'
        # For any type of URL other than a local file, just return the URL as-is.
        return url

    def __build_tarball(self) -> Path:
        bundle_path = self.tmp_dir / 'bundle.tar.bz2'
        with tarfile.open(bundle_path, 'w:bz2') as tf:
            # Add the iceberg warehouse dir (including the catalog)
            tf.add(self.tmp_dir / 'warehouse', arcname='warehouse')
            # Add the media files
            for src_file, dest_name in self.media_files.items():
                tf.add(src_file, arcname=f'media/{dest_name}')
        return bundle_path

import io
import json
import shutil
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Iterator, Union

import more_itertools
import numpy as np
import pyarrow as pa
from pyiceberg.catalog import Catalog
from pyiceberg.catalog.sql import SqlCatalog

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.utils.arrow import _pt_to_pa
from pixeltable.utils.iceberg import sqlite_catalog


class TablePackager:

    table: pxt.Table
    tmp_dir: Path
    tmp_media_dir: Path
    iceberg_catalog: Catalog

    def __init__(self, table: pxt.Table) -> None:
        self.table = table
        self.tmp_dir = Path(Env.get().create_tmp_path())
        self.tmp_media_dir = self.tmp_dir / 'media'

    def package(self) -> Path:
        self.tmp_dir.mkdir()
        self.iceberg_catalog = sqlite_catalog(self.tmp_dir / 'warehouse')
        ancestors = [self.table] + self.table._bases
        for t in ancestors:
            self.__export_table(t)
        return self.tmp_dir

    def __export_table(self, t: pxt.Table) -> None:
        # Select only those columns that are defined in this table (columns inherited from ancestor
        # tables will be handled separately)
        # TODO: This is selecting only named columns; do we also want to preserve system columns such as errortype?
        col_refs = []
        for col_name, col in t._tbl_version.cols_by_name.items():
            if col.col_type.is_media_type():
                col_refs.append(t[col_name].fileurl)
            else:
                col_refs.append(t[col_name])
        df = t.select(*col_refs)
        namespace = self.__iceberg_namespace(t)
        self.iceberg_catalog.create_namespace_if_not_exists(namespace)
        arrow_schema = self.__to_iceberg_schema(df._schema)
        iceberg_tbl = self.iceberg_catalog.create_table(f'{namespace}.{t._name}', schema=arrow_schema)
        for pa_table in self.__to_pa_tables(df, arrow_schema):
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

    def __to_pa_tables(self, df: pxt.DataFrame, arrow_schema: pa.Schema, batch_size: int = 1_000) -> Iterator[pa.Table]:
        for rows in more_itertools.batched(self.__to_pa_rows(df), batch_size):
            cols = {col_name: [row[idx] for row in rows] for idx, col_name in enumerate(df._schema.keys())}
            cols['_rowid'] = [row[-2] for row in rows]
            cols['_v_min'] = [row[-1] for row in rows]
            yield pa.Table.from_pydict(cols, schema=arrow_schema)

    def __to_pa_rows(self, df: pxt.DataFrame) -> Iterator[list]:
        for row in df._exec():
            result = [self.__to_pa_value(val, col_type) for val, col_type in zip(row, df._schema.values())]
            result.append(row.rowid)
            result.append(row.v_min)
            yield result

    def __to_pa_value(self, val: Any, col_type: ts.ColumnType) -> Any:
        if col_type.is_array_type():
            assert isinstance(val, np.ndarray)
            arr = io.BytesIO()
            np.save(arr, val)
            return arr.getvalue()
        if col_type.is_json_type():
            return json.dumps(val)  # Export JSON as strings
        if col_type.is_media_type():
            assert isinstance(val, str)
            return self.__process_media_url(val)
        return val

    def __process_media_url(self, url: str) -> str:
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.scheme == 'file':
            # It's the URL of a local file. Make a copy of the file in the archive location and return
            # a pxt:// URI that identifies the copied file.
            path = Path(parsed_url.path)
            id_hex = uuid.uuid4().hex
            new_filename = f'{id_hex}.{path.suffix}'
            new_path_parent = self.tmp_media_dir / id_hex[:2] / id_hex[:4]
            new_path_parent.mkdir(parents=True, exist_ok=True)
            new_path = new_path_parent / new_filename
            shutil.copyfile(path, new_path)
            return f'pxtmedia://{id_hex[:2]}/{id_hex[:4]}/{new_filename}'
        # For any type of URL other than a local file, just return the URL as-is.
        return url

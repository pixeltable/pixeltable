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
from pixeltable.utils.arrow import _pt_to_pa


def export_iceberg(table: pxt.Table, iceberg_catalog: Catalog) -> None:
    ancestors = [table] + table._bases
    for t in ancestors:
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
        namespace = __iceberg_namespace(t)
        iceberg_catalog.create_namespace_if_not_exists(namespace)
        arrow_schema = __to_arrow_schema_iceberg(df._schema)
        iceberg_tbl = iceberg_catalog.create_table(f'{namespace}.{t._name}', schema=arrow_schema)
        for pa_table in __to_pa_tables(df, arrow_schema):
            iceberg_tbl.append(pa_table)


def sqlite_catalog(iceberg_path: Union[str, Path]) -> SqlCatalog:
    if isinstance(iceberg_path, str):
        iceberg_path = Path(iceberg_path)
    iceberg_path.mkdir(exist_ok=True)
    return SqlCatalog('default', uri=f'sqlite:///{iceberg_path}/catalog.db', warehouse=f'file://{iceberg_path}')


def __iceberg_namespace(table: pxt.Table) -> str:
    """Iceberg tables must have a namespace, so we prepend `pxt` to the table path."""
    parent_path = table._parent._path
    if len(parent_path) == 0:
        return 'pxt'
    else:
        return f'pxt.{parent_path}'


def __to_arrow_schema_iceberg(pxt_schema: dict[str, ts.ColumnType]) -> pa.Schema:
    entries = [(name, __to_arrow_type_iceberg(col_type)) for name, col_type in pxt_schema.items()]
    entries.append(('_rowid', pa.list_(pa.int64())))
    entries.append(('_v_min', pa.int64()))
    return pa.schema(entries)  # type: ignore[arg-type]


def __to_arrow_type_iceberg(col_type: ts.ColumnType) -> pa.DataType:
    if col_type.is_array_type():
        return pa.binary()
    if col_type.is_media_type():
        return pa.string()
    return _pt_to_pa.get(col_type.__class__)


def __to_pa_tables(df: pxt.DataFrame, arrow_schema: pa.Schema, batch_size: int = 1_000) -> Iterator[pa.Table]:
    for rows in more_itertools.batched(__to_pa_rows(df), batch_size):
        cols = {col_name: [row[idx] for row in rows] for idx, col_name in enumerate(df._schema.keys())}
        cols['_rowid'] = [row[-2] for row in rows]
        cols['_v_min'] = [row[-1] for row in rows]
        yield pa.Table.from_pydict(cols, schema=arrow_schema)


def __to_pa_rows(df: pxt.DataFrame) -> Iterator[list]:
    for row in df._exec():
        result = [__to_pa_value(val, col_type) for val, col_type in zip(row, df._schema.values())]
        result.append(row.rowid)
        result.append(row.v_min)
        yield result


def __to_pa_value(val: Any, col_type: ts.ColumnType) -> Any:
    if col_type.is_array_type():
        assert isinstance(val, np.ndarray)
        arr = io.BytesIO()
        np.save(arr, val)
        return arr.getvalue()
    if col_type.is_json_type():
        return json.dumps(val)  # Export JSON as strings
    if col_type.is_media_type():
        assert isinstance(val, str)
        return __process_media_url(val)
    return val


def __process_media_url(url: str) -> str:
    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme == 'file':
        # It's the URL of a local file. Make a copy of the file in the archive location and return
        # a pxt:// URI that identifies the copied file.
        path = Path(parsed_url.path)
        new_filename = f'{uuid.uuid4().hex}.{path.suffix}'
        new_path = ...
        shutil.copyfile(path, new_path)
        return f'pxt://username/datapath/_media/{new_filename}'
    # For any type of URL other than a local file, just return the URL as-is.
    return url

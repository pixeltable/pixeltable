import datetime
import io
import json
from typing import Any, Iterator, Optional, Union

import more_itertools
import numpy as np
import pyarrow as pa

import pixeltable as pxt
import pixeltable.type_system as ts

_pa_to_pt: dict[pa.DataType, ts.ColumnType] = {
    pa.string(): ts.StringType(nullable=True),
    pa.bool_(): ts.BoolType(nullable=True),
    pa.uint8(): ts.IntType(nullable=True),
    pa.int8(): ts.IntType(nullable=True),
    pa.uint32(): ts.IntType(nullable=True),
    pa.uint64(): ts.IntType(nullable=True),
    pa.int32(): ts.IntType(nullable=True),
    pa.int64(): ts.IntType(nullable=True),
    pa.float32(): ts.FloatType(nullable=True),
}

_pt_to_pa: dict[type[ts.ColumnType], pa.DataType] = {
    ts.StringType: pa.string(),
    ts.TimestampType: pa.timestamp('us', tz=datetime.timezone.utc),  # postgres timestamp is microseconds
    ts.BoolType: pa.bool_(),
    ts.IntType: pa.int64(),
    ts.FloatType: pa.float32(),
    ts.ArrayType: pa.binary(),
    ts.JsonType: pa.string(),
    ts.ImageType: pa.binary(),  # inline image
    ts.AudioType: pa.string(),  # path
    ts.VideoType: pa.string(),  # path
    ts.DocumentType: pa.string(),  # path
}


def to_pixeltable_type(arrow_type: pa.DataType) -> Optional[ts.ColumnType]:
    """Convert a pyarrow DataType to a pixeltable ColumnType if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if isinstance(arrow_type, pa.TimestampType):
        return ts.TimestampType(nullable=True)
    elif arrow_type in _pa_to_pt:
        return _pa_to_pt[arrow_type]
    elif isinstance(arrow_type, pa.FixedShapeTensorType):
        dtype = to_pixeltable_type(arrow_type.value_type)
        if dtype is None:
            return None
        return ts.ArrayType(shape=arrow_type.shape, dtype=dtype)
    else:
        return None


def to_arrow_type(pixeltable_type: ts.ColumnType) -> Optional[pa.DataType]:
    """Convert a pixeltable DataType to a pyarrow datatype if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if pixeltable_type.__class__ in _pt_to_pa:
        return _pt_to_pa[pixeltable_type.__class__]
    else:
        return None


def to_pixeltable_schema(arrow_schema: pa.Schema) -> dict[str, ts.ColumnType]:
    return {field.name: to_pixeltable_type(field.type) for field in arrow_schema}


def to_arrow_schema(pixeltable_schema: dict[str, ts.ColumnType], include_rowid: bool = False) -> pa.Schema:
    entries = [(name, to_arrow_type(typ)) for name, typ in pixeltable_schema.items()]
    if include_rowid:
        entries.append(('_rowid', pa.list_(pa.int64())))
        entries.append(('_v_min', pa.int64()))
    return pa.schema(entries)  # type: ignore[misc]


def to_pydict(batch: Union[pa.Table, pa.RecordBatch]) -> dict[str, Union[list, np.ndarray]]:
    """Convert a RecordBatch to a dictionary of lists, unlike pa.lib.RecordBatch.to_pydict,
    this function will not convert numpy arrays to lists, and will preserve the original numpy dtype.
    """
    out: dict[str, Union[list, np.ndarray]] = {}
    for k, name in enumerate(batch.schema.names):
        col = batch.column(k)
        if isinstance(col.type, pa.FixedShapeTensorType):
            # treat array columns as numpy arrays to easily preserve numpy type
            out[name] = col.to_numpy(zero_copy_only=False)  # type: ignore[call-arg]
        else:
            # for the rest, use pydict to preserve python types
            out[name] = col.to_pylist()

    return out


def to_pa_tables(df: pxt.DataFrame, arrow_schema: pa.Schema, batch_size: int = 1_000, include_rowid: bool = False) -> Iterator[pa.Table]:
    for rows in more_itertools.batched(__to_pa_rows(df, include_rowid), batch_size):
        cols = {col_name: [row[idx] for row in rows] for idx, col_name in enumerate(df._schema.keys())}
        if include_rowid:
            cols['_rowid'] = [row[-2] for row in rows]
            cols['_v_min'] = [row[-1] for row in rows]
        yield pa.Table.from_pydict(cols, schema=arrow_schema)


def __to_pa_rows(df: pxt.DataFrame, include_rowid: bool = False) -> Iterator[list]:
    for row in df._exec():
        result = [__to_pa_value(val, col_type) for val, col_type in zip(row, df._schema.values())]
        if include_rowid:
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
    return val


def iter_tuples(batch: Union[pa.Table, pa.RecordBatch]) -> Iterator[dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        yield {col_name: values[i] for col_name, values in pydict.items()}

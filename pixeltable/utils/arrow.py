import datetime
from typing import Any, Iterator, Optional, Union

import numpy as np
import pyarrow as pa

import pixeltable.type_system as ts

PA_TO_PXT_TYPES: dict[pa.DataType, ts.ColumnType] = {
    pa.string(): ts.StringType(nullable=True),
    pa.large_string(): ts.StringType(nullable=True),
    pa.timestamp('us', tz=datetime.timezone.utc): ts.TimestampType(nullable=True),
    pa.bool_(): ts.BoolType(nullable=True),
    pa.int8(): ts.IntType(nullable=True),
    pa.int16(): ts.IntType(nullable=True),
    pa.int32(): ts.IntType(nullable=True),
    pa.int64(): ts.IntType(nullable=True),
    pa.uint8(): ts.IntType(nullable=True),
    pa.uint16(): ts.IntType(nullable=True),
    pa.uint32(): ts.IntType(nullable=True),
    pa.uint64(): ts.IntType(nullable=True),
    pa.float32(): ts.FloatType(nullable=True),
    pa.float64(): ts.FloatType(nullable=True),
    pa.date32(): ts.DateType(nullable=True),
    pa.date64(): ts.DateType(nullable=True),
    pa.binary(): None,  # cannot import binary (inline image)
}

PXT_TO_PA_TYPES: dict[type[ts.ColumnType], pa.DataType] = {
    ts.StringType: pa.string(),
    ts.TimestampType: pa.timestamp('us', tz=datetime.timezone.utc),  # postgres timestamp is microseconds
    ts.DateType: pa.date32(),  # This could be date64
    ts.BoolType: pa.bool_(),
    ts.IntType: pa.int64(),
    ts.FloatType: pa.float32(),
    ts.JsonType: pa.string(),  # TODO(orm) pa.struct() is possible
    ts.ImageType: pa.binary(),  # inline image
    ts.AudioType: pa.string(),  # path
    ts.VideoType: pa.string(),  # path
    ts.DocumentType: pa.string(),  # path
}


def to_pixeltable_type(arrow_type: pa.DataType, nullable: bool) -> Optional[ts.ColumnType]:
    """Convert a pyarrow DataType to a pixeltable ColumnType if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if isinstance(arrow_type, pa.TimestampType):
        return ts.TimestampType(nullable=nullable)
    elif arrow_type in PA_TO_PXT_TYPES:
        pt = PA_TO_PXT_TYPES[arrow_type]
        return pt.copy(nullable=nullable) if pt is not None else None
    elif isinstance(arrow_type, pa.FixedShapeTensorType):
        dtype = to_pixeltable_type(arrow_type.value_type, nullable)
        if dtype is None:
            return None
        return ts.ArrayType(shape=arrow_type.shape, dtype=dtype, nullable=nullable)
    else:
        return None


def to_arrow_type(pixeltable_type: ts.ColumnType) -> Optional[pa.DataType]:
    """Convert a pixeltable DataType to a pyarrow datatype if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if pixeltable_type.__class__ in PXT_TO_PA_TYPES:
        return PXT_TO_PA_TYPES[pixeltable_type.__class__]
    elif isinstance(pixeltable_type, ts.ArrayType):
        return pa.fixed_shape_tensor(pa.from_numpy_dtype(pixeltable_type.numpy_dtype()), pixeltable_type.shape)
    else:
        return None


def ar_infer_schema(
    arrow_schema: pa.Schema, schema_overrides: dict[str, Any], primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    """Convert a pyarrow Schema to a schema using pyarrow names and pixeltable types."""
    ar_schema = {
        field.name: to_pixeltable_type(field.type, field.name not in primary_key)
        if field.name not in schema_overrides
        else schema_overrides[field.name]
        for field in arrow_schema
    }
    return ar_schema


def to_arrow_schema(pixeltable_schema: dict[str, Any]) -> pa.Schema:
    return pa.schema((name, to_arrow_type(typ)) for name, typ in pixeltable_schema.items())  # type: ignore[misc]


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


def iter_tuples(batch: Union[pa.Table, pa.RecordBatch]) -> Iterator[dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        yield {col_name: values[i] for col_name, values in pydict.items()}


def _ar_val_to_pxt_val(val: Any, pxt_type: ts.ColumnType) -> Any:
    """Convert a value to insertable format"""
    if val is None:
        return None
    if pxt_type.is_float_type():
        return float(val)
    elif pxt_type.is_int_type():
        return int(val)
    elif pxt_type.is_bool_type():
        return bool(val)
    elif pxt_type.is_string_type():
        return str(val)
    elif pxt_type.is_date_type():
        if isinstance(val, str):
            return datetime.date.fromisoformat(val)
        if isinstance(val, datetime.date):
            return val
    elif pxt_type.is_timestamp_type():
        if isinstance(val, str):
            return datetime.datetime.fromisoformat(val)
        if isinstance(val, datetime.datetime):
            return val
    elif pxt_type.is_array_type():
        return pxt_type.create_literal(val)
    raise ValueError(f'Unsupported type {pxt_type} for value {val}')


def iter_tuples2(
    batch: Union[pa.Table, pa.RecordBatch], col_mapping: Optional[dict[str, str]], schema: dict[str, ts.ColumnType]
) -> Iterator[dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        # Convert a row to insertable format
        yield {
            (pxt_name := col_mapping.get(col_name, col_name)): _ar_val_to_pxt_val(values[i], schema[pxt_name])
            for col_name, values in pydict.items()
        }

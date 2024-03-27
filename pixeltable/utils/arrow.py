import logging
from typing import Any, Dict, Iterable, Iterator, Optional

import pyarrow as pa

import pixeltable.type_system as ts

_logger = logging.getLogger(__name__)

_pa_to_pt: Dict[pa.DataType, ts.ColumnType] = {
    pa.string(): ts.StringType(nullable=True),
    pa.timestamp('us'): ts.TimestampType(nullable=True),
    pa.bool_(): ts.BoolType(nullable=True),
    pa.uint8(): ts.IntType(nullable=True),
    pa.int8(): ts.IntType(nullable=True),
    pa.uint32(): ts.IntType(nullable=True),
    pa.uint64(): ts.IntType(nullable=True),
    pa.int32(): ts.IntType(nullable=True),
    pa.int64(): ts.IntType(nullable=True),
    pa.float32(): ts.FloatType(nullable=True),
}

_pt_to_pa: Dict[ts.ColumnType, pa.DataType] = {
    ts.StringType: pa.string(),
    ts.TimestampType: pa.timestamp('us'),  # postgres timestamp is microseconds
    ts.BoolType: pa.bool_(),
    ts.IntType: pa.int64(),
    ts.FloatType: pa.float32(),
    ts.JsonType: pa.string(),  # TODO(orm) pa.struct() is possible
    ts.ImageType: pa.binary(),  # inline image
    ts.AudioType: pa.string(),  # path
    ts.VideoType: pa.string(),  # path
    ts.DocumentType: pa.string(),  # path
}


def to_pixeltable_type(arrow_type: pa.DataType) -> Optional[ts.ColumnType]:
    """Convert a pyarrow DataType to a pixeltable ColumnType if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if arrow_type in _pa_to_pt:
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
    elif isinstance(pixeltable_type, ts.ArrayType):
        return pa.fixed_shape_tensor(pa.from_numpy_dtype(pixeltable_type.numpy_dtype()), pixeltable_type.shape)
    else:
        return None


def to_pixeltable_schema(arrow_schema: pa.Schema) -> Dict[str, ts.ColumnType]:
    return {field.name: to_pixeltable_type(field.type) for field in arrow_schema}


def to_arrow_schema(pixeltable_schema: Dict[str, Any]) -> pa.Schema:
    return pa.schema((name, to_arrow_type(typ)) for name, typ in pixeltable_schema.items())


def to_pydict(batch: pa.RecordBatch) -> Dict[str, Iterable[Any]]:
    """Convert a RecordBatch to a dictionary of lists, unlike pa.lib.RecordBatch.to_pydict,
    this function will not convert numpy arrays to lists, and will preserve the original numpy dtype.
    """
    out = {}
    for k, name in enumerate(batch.schema.names):
        col = batch.column(k)
        if isinstance(col.type, pa.FixedShapeTensorType):
            # treat array columns as numpy arrays to easily preserve numpy type
            out[name] = col.to_numpy(zero_copy_only=False)
        else:
            # for the rest, use pydict to preserve python types
            out[name] = col.to_pylist()

    return out


def iter_tuples(batch: pa.RecordBatch) -> Iterator[Dict[str, Any]]:
    """Convert a RecordBatch to an iterator of dictionaries. also works with pa.Table and pa.RowGroup"""
    pydict = to_pydict(batch)
    assert len(pydict) > 0, 'empty record batch'
    for _, v in pydict.items():
        batch_size = len(v)
        break

    for i in range(batch_size):
        yield {col_name: values[i] for col_name, values in pydict.items()}

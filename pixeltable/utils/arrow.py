import datetime
import io
import uuid
from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
import PIL.Image
import pyarrow as pa

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

if TYPE_CHECKING:
    import pixeltable as pxt

PA_TO_PXT_TYPES: dict[pa.DataType, ts.ColumnType] = {
    pa.string(): ts.StringType(nullable=True),
    pa.large_string(): ts.StringType(nullable=True),
    pa.timestamp('us', tz='UTC'): ts.TimestampType(nullable=True),
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
    pa.uuid(): ts.UUIDType(nullable=True),
    pa.binary(): ts.BinaryType(nullable=True),
}

PXT_TO_PA_TYPES: dict[type[ts.ColumnType], pa.DataType] = {
    ts.StringType: pa.string(),
    ts.TimestampType: pa.timestamp('us', tz='UTC'),  # postgres timestamp is microseconds
    ts.DateType: pa.date32(),  # This could be date64
    ts.UUIDType: pa.uuid(),
    ts.BoolType: pa.bool_(),
    ts.IntType: pa.int64(),
    ts.FloatType: pa.float32(),
    ts.BinaryType: pa.binary(),
    ts.ImageType: pa.binary(),  # inline image
    ts.AudioType: pa.string(),  # path
    ts.VideoType: pa.string(),  # path
    ts.DocumentType: pa.string(),  # path
    # ts.JsonType is ommitted, because mapping to pa.StructType requires schema
}


def to_pixeltable_type(arrow_type: pa.DataType, nullable: bool) -> ts.ColumnType | None:
    """Convert a pyarrow DataType to a pixeltable ColumnType if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if isinstance(arrow_type, pa.TimestampType):
        return ts.TimestampType(nullable=nullable)
    elif isinstance(arrow_type, pa.StructType):
        return ts.JsonType(nullable=nullable)
    elif arrow_type in PA_TO_PXT_TYPES:
        pt = PA_TO_PXT_TYPES[arrow_type]
        return pt.copy(nullable=nullable) if pt is not None else None
    elif isinstance(arrow_type, pa.FixedShapeTensorType):
        dtype = to_pixeltable_type(arrow_type.value_type, nullable)
        if dtype is None:
            return None
        return ts.ArrayType(shape=tuple(arrow_type.shape), dtype=dtype, nullable=nullable)
    else:
        return None


def to_arrow_type(pixeltable_type: ts.ColumnType) -> pa.DataType | None:
    """Convert a pixeltable DataType to a pyarrow datatype if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if pixeltable_type.__class__ in PXT_TO_PA_TYPES:
        return PXT_TO_PA_TYPES[pixeltable_type.__class__]
    elif isinstance(pixeltable_type, ts.ArrayType):
        return pa.fixed_shape_tensor(pa.from_numpy_dtype(pixeltable_type.dtype), pixeltable_type.shape)
    else:
        return None


def to_pxt_schema(
    arrow_schema: pa.Schema, schema_overrides: dict[str, Any], primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    """Convert a pyarrow Schema to a schema using pyarrow names and pixeltable types."""
    pxt_schema = {
        field.name: to_pixeltable_type(field.type, field.name not in primary_key)
        if field.name not in schema_overrides
        else schema_overrides[field.name]
        for field in arrow_schema
    }
    return pxt_schema


def _to_record_batch(column_vals: dict[str, list[Any]], schema: pa.Schema) -> pa.RecordBatch:
    import pyarrow as pa

    pa_arrays: list[pa.Array] = []
    for field in schema:
        if isinstance(field.type, pa.FixedShapeTensorType):
            stacked_arr = np.stack(column_vals[field.name])
            pa_arrays.append(pa.FixedShapeTensorArray.from_numpy_ndarray(stacked_arr))
        else:
            pa_array = cast(pa.Array, pa.array(column_vals[field.name]))
            pa_arrays.append(pa_array)
    return pa.RecordBatch.from_arrays(pa_arrays, schema=schema)


def to_record_batches(query: 'pxt.Query', batch_size_bytes: int) -> Iterator[pa.RecordBatch]:
    arrow_schema: pa.Schema | None = None  # initialized after first batch, when we have data to infer struct schemas
    batch_columns: dict[str, list[Any]] = {k: [] for k in query.schema}
    current_byte_estimate = 0
    num_batch_rows = 0
    json_val_size: dict[str, int] = {}  # key: col_name, value: average size of corresponding pa.struct
    json_batch_size: dict[str, int] = {}  # per-column cumulative size of json/struct values for first batch

    def create_arrow_schema() -> None:
        nonlocal arrow_schema
        if arrow_schema is not None:
            return
        pa_column_types: dict[str, pa.DataType] = {}
        for col_name, col_type in query.schema.items():
            if col_type.is_json_type():
                pa_type = pa.infer_type(batch_columns[col_name], mask=None)
                pa_column_types[col_name] = pa_type
            else:
                pa_column_types[col_name] = to_arrow_type(col_type)
        arrow_schema = pa.schema(pa_column_types.items())

    # TODO: in order to avoid having to deal with ExprEvalError here, ResultSet should be an iterator
    # over _exec()
    try:
        for data_row in query._exec():
            num_batch_rows += 1
            for (col_name, col_type), e in zip(query.schema.items(), query._select_list_exprs):
                val = data_row[e.slot_idx]
                val_size_bytes: int
                if val is None:
                    batch_columns[col_name].append(val)
                    continue

                assert val is not None
                if col_type.is_image_type():
                    # images get inlined into the parquet file
                    if data_row.file_paths[e.slot_idx] is not None:
                        # if there is a file, read directly to preserve information
                        with open(data_row.file_paths[e.slot_idx], 'rb') as f:
                            val = f.read()
                    elif isinstance(val, PIL.Image.Image):
                        # no file available: save as png
                        buf = io.BytesIO()
                        val.save(buf, format='png')
                        val = buf.getvalue()
                    else:
                        raise excs.Error(f'unknown image type {type(val)}')
                    val_size_bytes = len(val)
                elif col_type.is_string_type():
                    val_size_bytes = len(val)
                elif col_type.is_uuid_type():
                    # pa.uuid() uses fixed_size_binary(16) as storage type
                    val = val.bytes  # Convert UUID to 16-byte binary for arrow
                    val_size_bytes = len(val)
                elif col_type.is_binary_type():
                    val_size_bytes = len(val)
                elif col_type.is_media_type():
                    assert data_row.file_paths[e.slot_idx] is not None
                    val = data_row.file_paths[e.slot_idx]
                    val_size_bytes = len(val)
                elif col_type.is_json_type():
                    if col_name not in json_val_size:
                        val_size_bytes = pa.array([val]).nbytes
                        json_batch_size[col_name] = json_batch_size.get(col_name, 0) + val_size_bytes
                    else:
                        val_size_bytes = json_val_size[col_name]
                elif col_type.is_array_type():
                    val_size_bytes = val.nbytes
                elif col_type.is_int_type() or col_type.is_float_type():
                    val_size_bytes = 8
                elif col_type.is_bool_type():
                    val_size_bytes = 1
                elif col_type.is_date_type():
                    val_size_bytes = 4
                elif col_type.is_timestamp_type():
                    val = val.astimezone(datetime.timezone.utc)
                    val_size_bytes = 8
                else:
                    raise excs.Error(f'unknown type {col_type} for {col_name}')

                batch_columns[col_name].append(val)
                current_byte_estimate += val_size_bytes

            if current_byte_estimate > batch_size_bytes and num_batch_rows > 0:
                if arrow_schema is None:
                    # this is the first batch
                    json_val_size = {
                        col_name: json_batch_size[col_name] // num_batch_rows for col_name in json_batch_size
                    }
                create_arrow_schema()
                record_batch = _to_record_batch(batch_columns, arrow_schema)
                yield record_batch
                batch_columns = {k: [] for k in query.schema}
                current_byte_estimate = 0
                num_batch_rows = 0

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)

    if num_batch_rows > 0:
        create_arrow_schema()
        record_batch = _to_record_batch(batch_columns, arrow_schema)
        yield record_batch


def to_pydict(batch: pa.Table | pa.RecordBatch) -> dict[str, list | np.ndarray]:
    """Convert a RecordBatch to a dictionary of lists, unlike pa.lib.RecordBatch.to_pydict,
    this function will not convert numpy arrays to lists, and will preserve the original numpy dtype.
    """
    out: dict[str, list | np.ndarray] = {}
    for k, name in enumerate(batch.schema.names):
        col = batch.column(k)
        if isinstance(col.type, pa.FixedShapeTensorType):
            # treat array columns as numpy arrays to easily preserve numpy type
            out[name] = col.to_numpy(zero_copy_only=False)
        else:
            # for the rest, use pydict to preserve python types
            out[name] = col.to_pylist()

    return out


def iter_tuples(batch: pa.Table | pa.RecordBatch) -> Iterator[dict[str, Any]]:
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
    elif pxt_type.is_uuid_type():
        if isinstance(val, uuid.UUID):
            return val
        if isinstance(val, bytes):
            return uuid.UUID(bytes=val)
        return uuid.UUID(val)
    elif pxt_type.is_binary_type():
        assert isinstance(val, bytes)
        return val
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
    elif pxt_type.is_json_type():
        return val
    raise ValueError(f'Unsupported type {pxt_type} for value {val}')


def iter_tuples2(
    batch: pa.Table | pa.RecordBatch, col_mapping: dict[str, str] | None, schema: dict[str, ts.ColumnType]
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

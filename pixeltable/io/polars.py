import datetime as dt
from typing import Any, Optional

import polars as pl
import polars.datatypes as pl_types

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

PL_TO_PXT_TYPES: dict[pl.DataType, ts.ColumnType] = {
    pl_types.String(): ts.StringType(nullable=True),
    pl_types.Utf8(): ts.StringType(nullable=True),
    pl_types.Categorical(): ts.StringType(nullable=True),
    pl_types.Int8(): ts.IntType(nullable=True),
    pl_types.Int16(): ts.IntType(nullable=True),
    pl_types.Int32(): ts.IntType(nullable=True),
    pl_types.Int64(): ts.IntType(nullable=True),
    pl_types.UInt8(): ts.IntType(nullable=True),
    pl_types.UInt16(): ts.IntType(nullable=True),
    pl_types.UInt32(): ts.IntType(nullable=True),
    pl_types.UInt64(): ts.IntType(nullable=True),
    pl_types.Float32(): ts.FloatType(nullable=True),
    pl_types.Float64(): ts.FloatType(nullable=True),
    pl_types.Boolean(): ts.BoolType(nullable=True),
    pl_types.Date(): ts.DateType(nullable=True),
}

PXT_TO_PL_TYPES2: dict[ts.ColumnType.Type, pl_types.DataType] = {
    ts.ColumnType.Type.STRING: pl_types.String(),
    ts.ColumnType.Type.INT: pl_types.Int64(),
    ts.ColumnType.Type.FLOAT: pl_types.Float32(),
    ts.ColumnType.Type.BOOL: pl_types.Boolean(),
    ts.ColumnType.Type.JSON: pl_types.String(),  # TODO pl_types.struct() is possible
    ts.ColumnType.Type.IMAGE: pl_types.Binary(),  # inline image
    ts.ColumnType.Type.VIDEO: pl_types.String(),  # path
    ts.ColumnType.Type.AUDIO: pl_types.String(),  # path
    ts.ColumnType.Type.DOCUMENT: pl_types.String(),  # path
    ts.ColumnType.Type.DATE: pl_types.Date(),
}

# Note on polars Array and List types relative to Pixeltable Array types:
# - Polars types can be nested, so Arrays and Lists can occur within each other.
# - Polars Array types always have fixed 1d shape. Nesting can create multi-d shapes.
# - The conversion from a Pixeltable Array type to a polars Array type can result in
#   a mix of polars Array and List types.


def pxt_array_type_to_pl_array_type(pxt_type: ts.ArrayType) -> pl_types.DataType:
    """Create a polars Array or List type from a Pixeltable ArrayType.
    This handles nested arrays by creating nested polars types.
    """
    print(f'Converting array type: {pxt_type}, {pxt_type.dtype}, {pxt_type.shape}')
    inner_type = PXT_TO_PL_TYPES2.get(pxt_type.dtype, None)
    if inner_type is None:
        raise excs.Error(f'Cannot convert Pixeltable type {pxt_type.dtype} to polars type.')

    for dim in reversed(pxt_type.shape):
        if dim is None:
            inner_type = pl_types.List(inner_type)
        else:
            inner_type = pl_types.Array(inner=inner_type, shape=(dim,))
    print(f'Created polars type: {inner_type}')
    return inner_type


def pxt_to_pl_type(pxt_type: ts.ColumnType) -> Optional[pl.DataType]:
    """Convert a pixeltable DataType to a polars datatype if one is defined.
    Returns None if no conversion is currently implemented.
    """
    if pxt_type.type_enum in PXT_TO_PL_TYPES2:
        return PXT_TO_PL_TYPES2[pxt_type.type_enum]  # __class__]
    elif isinstance(pxt_type, ts.TimestampType):
        return pl_types.Datetime(time_unit='us', time_zone=Env.get().default_time_zone)
    elif isinstance(pxt_type, ts.ArrayType):
        return pxt_array_type_to_pl_array_type(pxt_type)
    else:
        return None


def pxt_to_pl_schema(pixeltable_schema: dict[str, Any]) -> pl.Schema:
    pl_dict = {name: pxt_to_pl_type(typ) for name, typ in pixeltable_schema.items()}
    return pl.Schema(pl_dict.items())


def pl_array_type_to_pxt_array_type(pl_type: pl_types.Array | pl_types.List) -> Optional[ts.ArrayType]:
    """Create a Pixeltable ArrayType from a polars Array or List type.
    This handles nested arrays by creating nested Pixeltable types.
    """
    if not isinstance(pl_type, (pl_types.Array, pl_types.List)):
        return None

    shape: list[int | None] = []
    inner_type: pl_types.Array | pl_types.List | pl.DataTypeClass | pl.DataType = pl_type

    while isinstance(inner_type, (pl_types.Array, pl_types.List)):
        if isinstance(inner_type, pl_types.Array):
            if len(inner_type.shape) != 1:
                return None
            shape.append(inner_type.shape[0])
            inner_type = inner_type.inner
        elif isinstance(inner_type, pl_types.List):
            shape.append(None)
            inner_type = inner_type.inner

    pxt_dtype = _get_pxt_type_for_pl_type(inner_type, nullable=True)
    if pxt_dtype is None:
        return None

    shape.reverse()
    return ts.ArrayType(dtype=pxt_dtype, shape=tuple(shape), nullable=True)


def _get_pxt_type_for_pl_type(pl_dtype: Any, nullable: bool) -> Optional[ts.ColumnType]:
    """Get Pixeltable type for basic polars data types.
    These are the only types which are supported as the inner type of a polars Array.
    """
    if pl_dtype in PL_TO_PXT_TYPES:
        pt = PL_TO_PXT_TYPES[pl_dtype]
        return pt.copy(nullable=nullable) if pt is not None else None

    elif isinstance(pl_dtype, pl_types.Enum):
        return ts.StringType(nullable=nullable)
    elif isinstance(pl_dtype, pl_types.Datetime):
        return ts.TimestampType(nullable=nullable)

    return None


def _pl_check_primary_key_values(df: 'pl.DataFrame', primary_key: list[str]) -> None:
    """Check that primary key columns don't contain null values."""
    for col_name in primary_key:
        null_count = df.get_column(col_name).null_count()
        if null_count > 0:
            raise excs.Error(f'Primary key column {col_name!r} cannot contain null values.')


def _pl_infer_schema(
    df: 'pl.DataFrame', schema_overrides: dict[str, ts.ColumnType], primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    """
    Infers a Pixeltable schema from a polars DataFrame.

    Returns:
        A Pixeltable schema dictionary.
    """
    pl_schema: dict[str, ts.ColumnType] = {}
    for col_name in df.columns:
        if col_name in schema_overrides:
            assert isinstance(schema_overrides[col_name], ts.ColumnType)
            pxt_type = schema_overrides[col_name]
        else:
            pl_dtype = df.get_column(col_name).dtype
            nullable = col_name not in (primary_key or [])
            pxt_type = _pl_dtype_to_pxt_type(pl_dtype, df.get_column(col_name), nullable)
        pl_schema[col_name] = pxt_type

    return pl_schema


def _pl_dtype_to_pxt_type(pl_type: 'pl.DataType', data_col: 'pl.Series', nullable: bool) -> ts.ColumnType:
    """
    Determines a Pixeltable ColumnType from a polars data type.

    Args:
        pl_type: A polars data type
        data_col: The actual data column for additional inference
        nullable: Whether the column can contain null values

    Returns:
        ts.ColumnType: A Pixeltable ColumnType
    """
    # Handle basic type mapping
    basic_type = _get_pxt_type_for_pl_type(pl_type, nullable)
    if basic_type is not None:
        return basic_type

    if isinstance(pl_type, (pl_types.Array, pl_types.List)):
        pxt_type = pl_array_type_to_pxt_array_type(pl_type)
        if pxt_type is not None:
            return pxt_type

    if isinstance(pl_type, (pl_types.List, pl_types.Struct, pl_types.Array)):
        return ts.JsonType(nullable=nullable)

    raise excs.Error(f'Could not infer Pixeltable type for polars column: {data_col.name} {pl_type}')


def _pl_row_to_pxt_row(
    row: dict[str, Any], schema: dict[str, ts.ColumnType], col_mapping: Optional[dict[str, str]]
) -> dict[str, Any]:
    """Convert a polars row to Pixeltable insertable format
    The polars DataFrame from which the rows are being sourced has already been converted to python types
    using the polars DataFrame method `to_dict`.
    This can be expensive in memory and time; if performance of this import becomes important,
    consider removing / optimizing this conversion.
    """
    pxt_row: dict[str, Any] = {}

    for col_name, val in row.items():
        pxt_name = col_mapping.get(col_name, col_name) if col_mapping else col_name
        pxt_type = schema[col_name]  # Use original column name for schema lookup

        # Handle null values
        if val is None:
            pxt_row[pxt_name] = None
            continue

        # Convert based on Pixeltable type
        nval: Any
        if pxt_type.is_float_type():
            nval = float(val)
        elif pxt_type.is_int_type():
            nval = int(val)
        elif pxt_type.is_bool_type():
            nval = bool(val)
        elif pxt_type.is_string_type() or (pxt_type.is_media_type() and isinstance(val, str)):
            # Process string types which overridden to be media types (e.g. file paths)
            nval = str(val)
        elif pxt_type.is_date_type():
            if not isinstance(val, dt.date):
                raise excs.Error(f'Expected date value for date column, got {type(val)}: {val}')
            nval = val
        elif pxt_type.is_timestamp_type():
            if not isinstance(val, dt.datetime):
                raise excs.Error(f'Expected datetime value for timestamp column, got {type(val)}: {val}')
            if val.tzinfo is not None and val.tzinfo.utcoffset(val) is not None:
                # Process timezone-aware datetime by converting to default timezone
                nval = val.astimezone(tz=Env.get().default_time_zone)
            else:
                # Process naive datetime by localizing to default timezone
                nval = val.replace(tzinfo=Env.get().default_time_zone)
        elif pxt_type.is_json_type():
            # For JSON types, keep the value as-is (lists, dicts, etc.)
            nval = val
        elif pxt_type.is_array_type():
            # polars Array types are converted into Python lists when exported
            if isinstance(val, list):
                import numpy as np

                nval = np.array(val)
            else:
                nval = val
        else:
            # For any other types, keep the value as-is
            nval = val

        pxt_row[pxt_name] = nval

    return pxt_row

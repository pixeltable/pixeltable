from typing import Any, Optional

import polars as pl
import polars.datatypes as pl_types

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

PL_TO_PXT_TYPES: dict[pl.DataType, ts.ColumnType] = {
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
    pl_types.String(): ts.StringType(nullable=True),
    pl_types.Utf8(): ts.StringType(nullable=True),
    pl_types.Categorical(): ts.StringType(nullable=True),
    pl_types.Boolean(): ts.BoolType(nullable=True),
    pl_types.Date(): ts.DateType(nullable=True),
}

# Function to map polars data types to Pixeltable types
def _get_pxt_type_for_pl_type(pl_dtype: pl.DataType | pl.DataTypeClass, nullable: bool) -> Optional[ts.ColumnType]:
    """Get Pixeltable type for basic polars data types.
    These are the only types which are supported as the inner type of a polars Array.
    """
    if pl_dtype in PL_TO_PXT_TYPES:
        pt = PL_TO_PXT_TYPES[pl_dtype]
        return pt.copy(nullable=nullable) if pt is not None else None

    # String types
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


def _pl_dtype_to_pxt_type(pl_dtype: 'pl.DataType', data_col: 'pl.Series', nullable: bool) -> ts.ColumnType:
    """
    Determines a Pixeltable ColumnType from a polars data type.

    Args:
        pl_dtype: A polars data type
        data_col: The actual data column for additional inference
        nullable: Whether the column can contain null values

    Returns:
        ts.ColumnType: A Pixeltable ColumnType
    """
    # Handle basic type mapping
    basic_type = _get_pxt_type_for_pl_type(pl_dtype, nullable)
    if basic_type is not None:
        return basic_type

    if isinstance(pl_dtype, pl_types.Array):
        pxt_dtype = _get_pxt_type_for_pl_type(pl_dtype.inner, nullable)
        if pxt_dtype is not None:
            return ts.ArrayType(shape=pl_dtype.shape, dtype=pxt_dtype, nullable=nullable)

    # Handle List/Array types
    if isinstance(pl_dtype, pl_types.List):
        return ts.JsonType(nullable=nullable)

    # Handle Struct types as JSON
    if isinstance(pl_dtype, pl_types.Struct):
        return ts.JsonType(nullable=nullable)

    raise excs.Error(f'Could not infer Pixeltable type for polars column: {data_col.name} (dtype: {pl_dtype})')


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
        elif pxt_type.is_string_type():
            nval = str(val)
        elif pxt_type.is_date_type():
            if hasattr(val, 'date'):
                nval = val.date()  # Extract date from datetime
            else:
                nval = val
        elif pxt_type.is_timestamp_type():
            if hasattr(val, 'astimezone'):
                # Handle timezone-aware timestamps
                nval = val.astimezone(Env.get().default_time_zone)
            else:
                # Handle naive timestamps - localize to default timezone
                import datetime as dt

                if isinstance(val, dt.datetime):
                    nval = val.replace(tzinfo=Env.get().default_time_zone)
                else:
                    nval = val
        elif pxt_type.is_json_type():
            # For JSON types, keep the value as-is (lists, dicts, etc.)
            nval = val
        elif pxt_type.is_array_type():
            # For array types, convert lists to numpy arrays as Pixeltable expects
            if isinstance(val, list):
                import numpy as np

                nval = np.array(val)
            elif hasattr(val, 'to_list'):
                # Handle polars Series objects from list columns
                import numpy as np

                nval = np.array(val.to_list())
            else:
                nval = val
        else:
            # For any other types, keep the value as-is
            nval = val

        pxt_row[pxt_name] = nval

    return pxt_row

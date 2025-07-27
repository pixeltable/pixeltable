from typing import TYPE_CHECKING, Any, Optional

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

if TYPE_CHECKING:
    import polars as pl


# Function to map polars data types to Pixeltable types
def _get_pxt_type_for_pl_type(pl_dtype: 'pl.DataType', nullable: bool) -> Optional[ts.ColumnType]:
    """Get Pixeltable type for a given polars data type."""
    import polars.datatypes as pl_types

    # Integer types
    if isinstance(
        pl_dtype,
        (
            pl_types.Int8,
            pl_types.Int16,
            pl_types.Int32,
            pl_types.Int64,
            pl_types.UInt8,
            pl_types.UInt16,
            pl_types.UInt32,
            pl_types.UInt64,
        ),
    ):
        return ts.IntType(nullable=nullable)
    # Float types
    elif isinstance(pl_dtype, (pl_types.Float32, pl_types.Float64)):
        return ts.FloatType(nullable=nullable)
    # Boolean type
    elif isinstance(pl_dtype, pl_types.Boolean):
        return ts.BoolType(nullable=nullable)
    # String types
    elif isinstance(pl_dtype, (pl_types.Utf8, pl_types.String)):
        return ts.StringType(nullable=nullable)
    # Date and time types
    elif isinstance(pl_dtype, pl_types.Date):
        return ts.DateType(nullable=nullable)
    elif isinstance(pl_dtype, pl_types.Datetime):
        return ts.TimestampType(nullable=nullable)
    elif isinstance(pl_dtype, (pl_types.Time, pl_types.Duration)):
        return ts.StringType(nullable=nullable)  # Pixeltable doesn't have Time type, use String

    return None


def _pl_check_primary_key_values(df: 'pl.DataFrame', primary_key: list[str]) -> None:
    """Check that primary key columns don't contain null values."""
    for col_name in primary_key:
        null_count = df.get_column(col_name).null_count()
        if null_count > 0:
            raise excs.Error(f'Primary key column `{col_name}` cannot contain null values.')


def pl_infer_schema(
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
    import polars.datatypes as pl_types

    # Handle basic type mapping
    basic_type = _get_pxt_type_for_pl_type(pl_dtype, nullable)
    if basic_type is not None:
        return basic_type

    # Handle List/Array types
    if isinstance(pl_dtype, pl_types.List):
        inner_type = pl_dtype.inner
        if isinstance(
            inner_type,
            (
                pl_types.Int8,
                pl_types.Int16,
                pl_types.Int32,
                pl_types.Int64,
                pl_types.UInt8,
                pl_types.UInt16,
                pl_types.UInt32,
                pl_types.UInt64,
            ),
        ):
            # Try to infer array shape from actual data
            if len(data_col) > 0:
                first_non_null = None
                for val in data_col:
                    if val is not None:
                        # polars returns Series objects for list elements
                        if hasattr(val, 'to_list'):
                            first_non_null = val.to_list()
                        elif isinstance(val, list):
                            first_non_null = val
                        else:
                            first_non_null = val
                        break

                if first_non_null is not None and hasattr(first_non_null, '__len__') and len(first_non_null) > 0:
                    return ts.ArrayType(shape=(None, len(first_non_null)), dtype=ts.IntType(), nullable=nullable)

            return ts.ArrayType(shape=(None, None), dtype=ts.IntType(), nullable=nullable)
        elif isinstance(inner_type, (pl_types.Float32, pl_types.Float64)):
            # Try to infer array shape from actual data for float arrays too
            if len(data_col) > 0:
                first_non_null = None
                for val in data_col:
                    if val is not None:
                        # polars returns Series objects for list elements
                        if hasattr(val, 'to_list'):
                            first_non_null = val.to_list()
                        elif isinstance(val, list):
                            first_non_null = val
                        else:
                            first_non_null = val
                        break

                if first_non_null is not None and hasattr(first_non_null, '__len__') and len(first_non_null) > 0:
                    return ts.ArrayType(shape=(None, len(first_non_null)), dtype=ts.FloatType(), nullable=nullable)

            return ts.ArrayType(shape=(None, None), dtype=ts.FloatType(), nullable=nullable)
        elif isinstance(inner_type, (pl_types.Utf8, pl_types.String)):
            # Handle List(String) - could be string arrays or mixed type converted to strings
            # Use JSON for flexibility since string arrays can have variable lengths
            return ts.JsonType(nullable=nullable)
        else:
            # For complex list types, use JSON
            return ts.JsonType(nullable=nullable)

    # Handle Struct types as JSON
    if isinstance(pl_dtype, pl_types.Struct):
        return ts.JsonType(nullable=nullable)

    # Handle Categorical as String
    if isinstance(pl_dtype, pl_types.Categorical):
        return ts.StringType(nullable=nullable)

    # Handle Object type by inspecting actual data
    if isinstance(pl_dtype, pl_types.Object):
        # Make a copy of the Series without null values for type inference
        non_null_data = data_col.drop_nulls()

        if len(non_null_data) == 0:
            # No non-null values; default to StringType
            return ts.StringType(nullable=nullable)

        # Convert to Python values for type inference
        sample_values = non_null_data.to_list()[:100]  # Sample first 100 values

        inferred_type = ts.ColumnType.infer_common_literal_type(sample_values)
        if inferred_type is None:
            # Fallback on StringType if everything else fails
            return ts.StringType(nullable=nullable)
        else:
            return inferred_type.copy(nullable=nullable)

    # Handle Null type
    if isinstance(pl_dtype, pl_types.Null):
        return ts.StringType(nullable=True)  # Default to nullable string for null columns

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

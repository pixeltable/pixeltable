import os
from typing import Any, Optional, Union

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

try:
    import polars as pl
    import polars.datatypes as pl_types
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False


# Mapping from Polars data types to Pixeltable types
_POLARS_TO_PXT_TYPE_MAP = {
    # Integer types
    pl_types.Int8: ts.IntType,
    pl_types.Int16: ts.IntType, 
    pl_types.Int32: ts.IntType,
    pl_types.Int64: ts.IntType,
    pl_types.UInt8: ts.IntType,
    pl_types.UInt16: ts.IntType,
    pl_types.UInt32: ts.IntType,
    pl_types.UInt64: ts.IntType,
    
    # Float types
    pl_types.Float32: ts.FloatType,
    pl_types.Float64: ts.FloatType,
    
    # Boolean type
    pl_types.Boolean: ts.BoolType,
    
    # String types
    pl_types.Utf8: ts.StringType,
    pl_types.String: ts.StringType,
    
    # Date and time types
    pl_types.Date: ts.DateType,
    pl_types.Datetime: ts.TimestampType,
    pl_types.Time: ts.StringType,  # Pixeltable doesn't have a Time type, use String
    pl_types.Duration: ts.StringType,  # Duration as string representation
    
    # Binary type
    pl_types.Binary: ts.StringType,  # Store as base64 string
}


def _check_polars_available() -> None:
    """Check if Polars is available and raise appropriate error if not."""
    if not POLARS_AVAILABLE:
        raise excs.Error(
            "Polars integration requires the 'polars' package. "
            "Install it with: pip install polars"
        )


def import_polars(
    tbl_name: str,
    df: 'pl.DataFrame',
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> pxt.Table:
    """Creates a new base table from a Polars
    [`DataFrame`](https://pola-rs.github.io/polars/py-polars/html/reference/dataframe/index.html), with the
    specified name. The schema of the table will be inferred from the DataFrame.

    The column names of the new table will be identical to those in the DataFrame, as long as they are valid
    Pixeltable identifiers. If a column name is not a valid Pixeltable identifier, it will be normalized according to
    the following procedure:
    - first replace any non-alphanumeric characters with underscores;
    - then, preface the result with the letter 'c' if it begins with a number or an underscore;
    - then, if there are any duplicate column names, suffix the duplicates with '_2', '_3', etc., in column order.

    Args:
        tbl_name: The name of the table to create.
        df: The Polars `DataFrame`.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
            name `name` will be given type `type`, instead of being inferred from the `DataFrame`. The keys in
            `schema_overrides` should be the column names of the `DataFrame` (whether or not they are valid
            Pixeltable identifiers).

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    _check_polars_available()
    
    return pxt.create_table(
        tbl_name,
        source=df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def import_polars_csv(
    tbl_name: str,
    source: Union[str, os.PathLike],
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from a CSV file using Polars. This is a convenience method and is equivalent
    to calling `import_polars(table_path, pl.read_csv(source, **kwargs), schema=schema)`.
    See the Polars documentation for [`read_csv`](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_csv.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    _check_polars_available()
    
    # Use Polars to read CSV, then create table from resulting DataFrame
    df = pl.read_csv(source, **kwargs)
    return import_polars(
        tbl_name,
        df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def import_polars_parquet(
    tbl_name: str,
    source: Union[str, os.PathLike],
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from a Parquet file using Polars. This is a convenience method and is equivalent
    to calling `import_polars(table_path, pl.read_parquet(source, **kwargs), schema=schema)`.
    See the Polars documentation for [`read_parquet`](https://pola-rs.github.io/polars/py-polars/html/reference/api/polars.read_parquet.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    _check_polars_available()
    
    # Use Polars to read Parquet, then create table from resulting DataFrame
    df = pl.read_parquet(source, **kwargs)
    return import_polars(
        tbl_name,
        df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def _pl_check_primary_key_values(df: 'pl.DataFrame', primary_key: list[str]) -> None:
    """Check that primary key columns don't contain null values."""
    for col_name in primary_key:
        null_count = df.get_column(col_name).null_count()
        if null_count > 0:
            raise excs.Error(f'Primary key column `{col_name}` cannot contain null values.')


def pl_infer_schema(
    df: 'pl.DataFrame', 
    schema_overrides: dict[str, ts.ColumnType], 
    primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    """
    Infers a Pixeltable schema from a Polars DataFrame.

    Returns:
        A Pixeltable schema dictionary.
    """
    _check_polars_available()
    
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
    Determines a Pixeltable ColumnType from a Polars data type.

    Args:
        pl_dtype: A Polars data type
        data_col: The actual data column for additional inference
        nullable: Whether the column can contain null values

    Returns:
        ts.ColumnType: A Pixeltable ColumnType
    """
    # Handle basic type mapping
    for pl_type, pxt_type_class in _POLARS_TO_PXT_TYPE_MAP.items():
        if isinstance(pl_dtype, pl_type):
            return pxt_type_class(nullable=nullable)
    
    # Handle List/Array types
    if isinstance(pl_dtype, pl_types.List):
        inner_type = pl_dtype.inner
        if isinstance(inner_type, (pl_types.Int8, pl_types.Int16, pl_types.Int32, pl_types.Int64,
                                   pl_types.UInt8, pl_types.UInt16, pl_types.UInt32, pl_types.UInt64)):
            # Try to infer array shape from actual data
            if len(data_col) > 0:
                first_non_null = None
                for val in data_col:
                    if val is not None:
                        first_non_null = val
                        break
                
                if first_non_null is not None and len(first_non_null) > 0:
                    return ts.ArrayType(shape=(None, len(first_non_null)), dtype=ts.IntType(), nullable=nullable)
            
            return ts.ArrayType(shape=(None, None), dtype=ts.IntType(), nullable=nullable)
        elif isinstance(inner_type, (pl_types.Float32, pl_types.Float64)):
            return ts.ArrayType(shape=(None, None), dtype=ts.FloatType(), nullable=nullable)
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
        # Drop null values for type inference
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
    
    raise excs.Error(f'Could not infer Pixeltable type for Polars column: {data_col.name} (dtype: {pl_dtype})')


def _pl_row_to_pxt_row(
    row: dict[str, Any], 
    schema: dict[str, ts.ColumnType], 
    col_mapping: Optional[dict[str, str]]
) -> dict[str, Any]:
    """Convert a Polars row to Pixeltable insertable format"""
    pxt_row: dict[str, Any] = {}
    
    for col_name, val in row.items():
        pxt_name = col_mapping.get(col_name, col_name) if col_mapping else col_name
        pxt_type = schema[pxt_name]
        
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
            else:
                nval = val
        else:
            # For any other types, keep the value as-is
            nval = val
            
        pxt_row[pxt_name] = nval
    
    return pxt_row 
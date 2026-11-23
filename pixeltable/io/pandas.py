from __future__ import annotations

import os
import typing
import uuid
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
from pandas._typing import DtypeObj  # For pandas dtype type hints
from pandas.api.types import is_datetime64_any_dtype, is_extension_array_dtype

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env

if typing.TYPE_CHECKING:
    import pixeltable as pxt


def import_pandas(
    tbl_name: str,
    df: pd.DataFrame,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    comment: str = '',
) -> pxt.Table:
    """Creates a new base table from a Pandas
    [`DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), with the
    specified name. The schema of the table will be inferred from the DataFrame.

    The column names of the new table will be identical to those in the DataFrame, as long as they are valid
    Pixeltable identifiers. If a column name is not a valid Pixeltable identifier, it will be normalized according to
    the following procedure:
    - first replace any non-alphanumeric characters with underscores;
    - then, preface the result with the letter 'c' if it begins with a number or an underscore;
    - then, if there are any duplicate column names, suffix the duplicates with '_2', '_3', etc., in column order.

    Args:
        tbl_name: The name of the table to create.
        df: The Pandas `DataFrame`.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
            name `name` will be given type `type`, instead of being inferred from the `DataFrame`. The keys in
            `schema_overrides` should be the column names of the `DataFrame` (whether or not they are valid
            Pixeltable identifiers).

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        tbl_name, source=df, schema_overrides=schema_overrides, primary_key=primary_key, comment=comment
    )


def import_excel(
    tbl_name: str,
    io: str | os.PathLike,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from an Excel (.xlsx) file. This is a convenience method and is
    equivalent to calling `import_pandas(table_path, pd.read_excel(io, *args, **kwargs), schema=schema)`.
    See the Pandas documentation for [`read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        tbl_name,
        source=io,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        comment=comment,
        extra_args=kwargs,
    )


def _df_check_primary_key_values(df: pd.DataFrame, primary_key: list[str]) -> None:
    for pd_name in primary_key:
        # This can be faster for large DataFrames
        has_nulls = df[pd_name].count() < len(df)
        if has_nulls:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Primary key column `{pd_name}` cannot contain null values.'
            )


def df_infer_schema(
    df: pd.DataFrame, schema_overrides: dict[str, ts.ColumnType], primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    """
    Infers a Pixeltable schema from a Pandas DataFrame.

    Returns:
        A tuple containing a Pixeltable schema and a list of primary key column names.
    """
    pd_schema: dict[str, ts.ColumnType] = {}
    for pd_name, pd_dtype in zip(df.columns, df.dtypes):
        if pd_name in schema_overrides:
            assert isinstance(schema_overrides[pd_name], ts.ColumnType)
            pxt_type = schema_overrides[pd_name]
        else:
            pxt_type = __pd_coltype_to_pxt_type(pd_dtype, df[pd_name], pd_name not in primary_key)
        pd_schema[pd_name] = pxt_type

    return pd_schema


def __pd_dtype_to_pxt_type(pd_dtype: DtypeObj, nullable: bool) -> ts.ColumnType | None:
    """
    Determines a pixeltable ColumnType from a pandas dtype

    Args:
        pd_dtype: A pandas dtype object

    Returns:
        ts.ColumnType: A pixeltable ColumnType
    """
    # Pandas extension arrays / types (Int64, boolean, string[pyarrow], etc.) are not directly
    # compatible with NumPy dtypes
    # The timezone-aware datetime64[ns, tz=] dtype is a pandas extension dtype
    if is_datetime64_any_dtype(pd_dtype):
        return ts.TimestampType(nullable=nullable)
    if isinstance(pd_dtype, pd.StringDtype):
        return ts.StringType(nullable=nullable)
    if is_extension_array_dtype(pd_dtype):
        # Nullable and pyarrow-backed numeric/boolean extension dtypes (Int64, Float64, boolean, ...)
        kind = getattr(pd_dtype, 'kind', None)
        if kind == 'b':
            return ts.BoolType(nullable=nullable)
        if kind in ('i', 'u'):
            return ts.IntType(nullable=nullable)
        if kind == 'f':
            return ts.FloatType(nullable=nullable)
        return None
    # Most other pandas dtypes are directly NumPy compatible
    assert isinstance(pd_dtype, np.dtype)
    return ts.ColumnType.from_np_dtype(pd_dtype, nullable)


def __pd_coltype_to_pxt_type(pd_dtype: DtypeObj, data_col: pd.Series, nullable: bool) -> ts.ColumnType:
    """
    Infers a Pixeltable type based on a pandas dtype.
    """
    pxttype = __pd_dtype_to_pxt_type(pd_dtype, nullable)
    if pxttype is not None:
        return pxttype

    if pd_dtype == np.object_:
        # The `object_` dtype can mean all sorts of things; see if we can infer the Pixeltable type
        # based on the actual data in `data_col`.
        # First drop any null values (they don't contribute to type inference).
        data_col = data_col.dropna()

        if len(data_col) == 0:
            # No non-null values; default to FloatType (the Pandas type of an all-NaN column)
            return ts.FloatType(nullable=nullable)

        inferred_type = ts.ColumnType.infer_common_literal_type(data_col)
        if inferred_type is None:
            # Fallback on StringType if everything else fails
            return ts.StringType(nullable=nullable)
        else:
            return inferred_type.copy(nullable=nullable)

    raise excs.RequestError(
        excs.ErrorCode.INVALID_TYPE, f'Could not infer Pixeltable type of column: {data_col.name} (dtype: {pd_dtype})'
    )


def _df_prepare_columns(
    df: pd.DataFrame, schema: dict[str, ts.ColumnType], col_mapping: dict[str, str]
) -> dict[str, Iterable[Any]]:
    """Prepare columns without materializing row dictionaries or modifying the source DataFrame."""
    return {col_mapping.get(name, name): _df_prepare_column(df[name], col_type) for name, col_type in schema.items()}


def _df_rows(prepared_columns: dict[str, Iterable[Any]], num_rows: int) -> Iterator[dict[str, Any]]:
    """Yield row dictionaries from prepared columns."""
    # Keep normalized timestamps in pandas storage; replace NaT only when constructing a row.
    columns = [
        (None if value is pd.NaT else value for value in column)
        if isinstance(column, pd.Series) and is_datetime64_any_dtype(column.dtype)
        else column
        for column in prepared_columns.values()
    ]
    # Include the row count so a DataFrame with no columns still supplies its empty rows.
    for row in zip(range(num_rows), *columns):
        yield dict(zip(prepared_columns, row[1:]))


def _df_prepare_column(data: pd.Series, col_type: ts.ColumnType) -> Iterable[Any]:
    dtype = data.dtype
    if isinstance(dtype, np.dtype):
        if col_type.is_int_type() and dtype.kind in ('i', 'u'):
            return data
        if col_type.is_float_type():
            if dtype.kind == 'f' and dtype.itemsize <= 8:
                return data
            if dtype.kind in ('b', 'i', 'u'):
                return data.astype(np.float64)
        if col_type.is_bool_type():
            if dtype.kind == 'b':
                return data
            if dtype.kind in ('i', 'u'):
                return data.astype(bool)

    if is_datetime64_any_dtype(dtype):
        try:
            if col_type.is_timestamp_type():
                tz = Env.get().default_time_zone
                return data.dt.tz_localize(tz) if data.dt.tz is None else data.dt.tz_convert(tz)
            if col_type.is_date_type():
                return data.dt.date.where(data.notna(), None)
        except Exception:
            # Preserve scalar conversion for datetime cases a pandas column operation cannot handle.
            pass

    return [_df_convert_value(val, col_type, str(data.name)) for val in data]


def _df_convert_value(val: Any, col_type: ts.ColumnType, col_name: str) -> Any:
    if val is None or val is pd.NA or val is pd.NaT:
        # Missing values in object and nullable extension columns; float('nan') is handled below.
        return None
    if col_type.is_float_type():
        return float(val)
    if isinstance(val, float) and np.isnan(val):
        # Float NaN remains a float value; in other column types it represents a missing cell.
        return None
    if col_type.is_int_type():
        if isinstance(val, (float, np.floating)) and not float(val).is_integer():
            raise excs.RequestError(
                excs.ErrorCode.INVALID_TYPE,
                f'Value {val!r} in column `{col_name}` cannot be converted to int without loss of precision',
            )
        return int(val)
    if col_type.is_bool_type():
        return bool(val)
    if col_type.is_string_type():
        return str(val)
    if col_type.is_date_type():
        return None if pd.isnull(val) else pd.Timestamp(val).date()
    if col_type.is_timestamp_type():
        if pd.isnull(val):
            return None
        tval = pd.Timestamp(val)
        if tval.tz is None:
            return tval.tz_localize(tz=Env.get().default_time_zone)
        return tval.astimezone(Env.get().default_time_zone)
    if col_type.is_uuid_type():
        if pd.isnull(val):
            return None
        return val if isinstance(val, uuid.UUID) else uuid.UUID(val)
    return val

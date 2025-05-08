import os
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas._typing import DtypeObj  # For pandas dtype type hints
from pandas.api.types import is_datetime64_any_dtype, is_extension_array_dtype

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env


def import_pandas(
    tbl_name: str,
    df: pd.DataFrame,
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
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
        tbl_name,
        source=df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def import_csv(
    tbl_name: str,
    filepath_or_buffer: Union[str, os.PathLike],
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from a csv file. This is a convenience method and is equivalent
    to calling `import_pandas(table_path, pd.read_csv(filepath_or_buffer, **kwargs), schema=schema)`.
    See the Pandas documentation for [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        tbl_name,
        source=filepath_or_buffer,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
        extra_args=kwargs,
    )


def import_excel(
    tbl_name: str,
    io: Union[str, os.PathLike],
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
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
        num_retained_versions=num_retained_versions,
        comment=comment,
        extra_args=kwargs,
    )


def _df_check_primary_key_values(df: pd.DataFrame, primary_key: list[str]) -> None:
    for pd_name in primary_key:
        # This can be faster for large DataFrames
        has_nulls = df[pd_name].count() < len(df)
        if has_nulls:
            raise excs.Error(f'Primary key column `{pd_name}` cannot contain null values.')


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
            pxt_type = schema_overrides[pd_name]
        else:
            pxt_type = __pd_coltype_to_pxt_type(pd_dtype, df[pd_name], pd_name not in primary_key)
        pd_schema[pd_name] = pxt_type

    return pd_schema


def __pd_dtype_to_pxt_type(pd_dtype: DtypeObj, nullable: bool) -> Optional[ts.ColumnType]:
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
    if is_extension_array_dtype(pd_dtype):
        return None
    # Most other pandas dtypes are directly NumPy compatible
    assert isinstance(pd_dtype, np.dtype)
    return ts.ArrayType.from_np_dtype(pd_dtype, nullable)


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

    raise excs.Error(f'Could not infer Pixeltable type of column: {data_col.name} (dtype: {pd_dtype})')


def _df_row_to_pxt_row(
    row: tuple[Any, ...], schema: dict[str, ts.ColumnType], col_mapping: Optional[dict[str, str]]
) -> dict[str, Any]:
    """Convert a row to insertable format"""
    pxt_row: dict[str, Any] = {}
    for val, (col_name, pxt_type) in zip(row[1:], schema.items()):
        pxt_name = col_mapping.get(col_name, col_name)
        nval: Any
        if pxt_type.is_float_type():
            nval = float(val)
        elif isinstance(val, float) and np.isnan(val):
            # pandas uses NaN for empty cells, even for types other than float;
            # for any type but a float, convert these to None
            nval = None
        elif pxt_type.is_int_type():
            nval = int(val)
        elif pxt_type.is_bool_type():
            nval = bool(val)
        elif pxt_type.is_string_type():
            nval = str(val)
        elif pxt_type.is_date_type():
            if pd.isnull(val):
                # pandas has the bespoke 'NaT' valud for a missing timestamp
                # This is not supported by postgres, and must be converted to None
                nval = None
            else:
                nval = pd.Timestamp(val).date()
        elif pxt_type.is_timestamp_type():
            if pd.isnull(val):
                # pandas has the bespoke 'NaT' value for a missing timestamp
                # This is not supported by postgres, and must be converted to None
                nval = None
            else:
                tval = pd.Timestamp(val)
                # pandas supports tz-aware and naive timestamps.
                if tval.tz is None:
                    nval = pd.Timestamp(tval).tz_localize(tz=Env.get().default_time_zone)
                else:
                    nval = tval.astimezone(Env.get().default_time_zone)
        else:
            nval = val
        pxt_row[pxt_name] = nval
    return pxt_row

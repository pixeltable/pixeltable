from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from pandas._typing import DtypeObj  # For pandas dtype type hints
from pandas.api.types import is_datetime64_any_dtype, is_extension_array_dtype

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import Table

from .utils import find_or_create_table, normalize_import_parameters, normalize_schema_names


def import_pandas(
    tbl_name: str,
    df: pd.DataFrame,
    *,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> pxt.Table:
    from pixeltable.io.globals import create_from_import
    return create_from_import(tbl_name, source=df, schema=schema_overrides, primary_key=primary_key, num_retained_versions=num_retained_versions, comment=comment)


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
    schema_overrides, primary_key = normalize_import_parameters(schema_overrides, primary_key)
    pd_schema = df_infer_schema(df, schema_overrides, primary_key)
    schema, pxt_pk, col_mapping = normalize_schema_names(pd_schema, primary_key, schema_overrides, False)

    _df_check_primary_key_values(df, primary_key)

    # Convert all rows to insertable format
    tbl_rows = [_df_row_to_pxt_row(row, pd_schema, col_mapping) for row in df.itertuples()]

    table = find_or_create_table(
        tbl_name, schema, primary_key=pxt_pk, num_retained_versions=num_retained_versions, comment=comment
    )
    table.insert(tbl_rows)
    return table


def import_csv(
    tbl_name: str,
    filepath_or_buffer,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs,
) -> pxt.Table:
    """
    Creates a new base table from a csv file. This is a convenience method and is equivalent
    to calling `import_pandas(table_path, pd.read_csv(filepath_or_buffer, **kwargs), schema=schema)`.
    See the Pandas documentation for [`read_csv`](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    df = pd.read_csv(filepath_or_buffer, **kwargs)
    return import_pandas(
        tbl_name,
        df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def import_excel(
    tbl_name: str,
    io,
    *args,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs,
) -> pxt.Table:
    """
    Creates a new base table from an Excel (.xlsx) file. This is a convenience method and is
    equivalent to calling `import_pandas(table_path, pd.read_excel(io, *args, **kwargs), schema=schema)`.
    See the Pandas documentation for [`read_excel`](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html)
    for more details.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    df = pd.read_excel(io, *args, **kwargs)
    return import_pandas(
        tbl_name,
        df,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def _df_check_primary_key_values(df: pd.DataFrame, primary_key: list[str]) -> None:
    for pd_name in primary_key:
        # This can be faster for large DataFrames
        has_nulls = df[pd_name].count() < len(df)
        if has_nulls:
            raise excs.Error(f'Primary key column `{pd_name}` cannot contain null values.')


def df_infer_schema(
    df: pd.DataFrame, schema_overrides: dict[str, pxt.ColumnType], primary_key: list[str]
) -> dict[str, pxt.ColumnType]:
    """
    Infers a Pixeltable schema from a Pandas DataFrame.

    Returns:
        A tuple containing a Pixeltable schema and a list of primary key column names.
    """
    pd_schema: dict[str, pxt.ColumnType] = {}
    for pd_name, pd_dtype in zip(df.columns, df.dtypes):
        if pd_name in schema_overrides:
            pxt_type = schema_overrides[pd_name]
        else:
            pxt_type = __pd_coltype_to_pxt_type(pd_dtype, df[pd_name], pd_name not in primary_key)
        pd_schema[pd_name] = pxt_type

    return pd_schema


"""
# Check if a datetime64[ns, UTC] dtype
def is_datetime_tz_utc(x: Any) -> bool:
    if isinstance(x, pd.Timestamp) and x.tzinfo is not None and str(x.tzinfo) == 'UTC':
        return True
    return pd.api.types.is_datetime64tz_dtype(x) and str(x).endswith('UTC]')
"""


def __pd_dtype_to_pxt_type(pd_dtype: DtypeObj, nullable: bool) -> Optional[pxt.ColumnType]:
    """
    Determines a pixeltable ColumnType from a pandas dtype

    Args:
        pd_dtype: A pandas dtype object

    Returns:
        pxt.ColumnType: A pixeltable ColumnType
    """
    # Pandas extension arrays / types (Int64, boolean, string[pyarrow], etc.) are not directly compatible with NumPy dtypes
    # The timezone-aware datetime64[ns, tz=] dtype is a pandas extension dtype
    if is_datetime64_any_dtype(pd_dtype):
        return pxt.TimestampType(nullable=nullable)
    if is_extension_array_dtype(pd_dtype):
        return None
    # Most other pandas dtypes are directly NumPy compatible
    assert isinstance(pd_dtype, np.dtype)
    return pxt.ArrayType.from_np_dtype(pd_dtype, nullable)


def __pd_coltype_to_pxt_type(pd_dtype: DtypeObj, data_col: pd.Series, nullable: bool) -> pxt.ColumnType:
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
            return pxt.FloatType(nullable=nullable)

        inferred_type = pxt.ColumnType.infer_common_literal_type(data_col)
        if inferred_type is None:
            # Fallback on StringType if everything else fails
            return pxt.StringType(nullable=nullable)
        else:
            return inferred_type.copy(nullable=nullable)

    raise excs.Error(f'Could not infer Pixeltable type of column: {data_col.name} (dtype: {pd_dtype})')


def _df_row_to_pxt_row(
    row: tuple[Any, ...], schema: dict[str, pxt.ColumnType], col_mapping: Optional[dict[str, str]]
) -> dict[str, Any]:
    """Convert a row to insertable format"""
    pxt_row: dict[str, Any] = {}
    for val, (col_name, pxt_type) in zip(row[1:], schema.items()):
        if pxt_type.is_float_type():
            val = float(val)
        elif isinstance(val, float) and np.isnan(val):
            # pandas uses NaN for empty cells, even for types other than float;
            # for any type but a float, convert these to None
            val = None
        elif pxt_type.is_int_type():
            val = int(val)
        elif pxt_type.is_bool_type():
            val = bool(val)
        elif pxt_type.is_string_type():
            val = str(val)
        elif pxt_type.is_timestamp_type():
            if pd.isnull(val):
                # pandas has the bespoke 'NaT' type for a missing timestamp; postgres is very
                # much not-ok with it. (But if we convert it to None and then load out the
                # table contents as a pandas DataFrame, it will correctly restore the 'NaT'!)
                val = None
            else:
                val = pd.Timestamp(val).to_pydatetime()
        pxt_name = col_name if col_mapping is None else col_mapping[col_name]
        pxt_row[pxt_name] = val
    return pxt_row

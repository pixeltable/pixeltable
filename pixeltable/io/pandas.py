import datetime
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts


def import_pandas(
    tbl_name: str, df: pd.DataFrame, *, schema_overrides: Optional[dict[str, pxt.ColumnType]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = ''
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
    if schema_overrides is None:
        schema_overrides = {}
    if primary_key is None:
        primary_key = []
    elif isinstance(primary_key, str):
        primary_key = [primary_key]

    schema, pxt_pk = __df_to_pxt_schema(df, schema_overrides, primary_key)
    tbl_rows = (dict(__df_row_to_pxt_row(row, schema)) for row in df.itertuples())
    table = pxt.create_table(tbl_name, schema, primary_key=pxt_pk, num_retained_versions=num_retained_versions, comment=comment)
    table.insert(tbl_rows)
    return table


def import_csv(
    tbl_name: str, filepath_or_buffer, schema_overrides: Optional[dict[str, ts.ColumnType]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs
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
    return import_pandas(tbl_name, df, schema_overrides=schema_overrides, primary_key=primary_key, num_retained_versions=num_retained_versions, comment=comment)


def import_excel(
    tbl_name: str, io, *args, schema_overrides: Optional[dict[str, ts.ColumnType]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs
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
    return import_pandas(tbl_name, df, schema_overrides=schema_overrides, primary_key=primary_key, num_retained_versions=num_retained_versions, comment=comment)


def __df_to_pxt_schema(
    df: pd.DataFrame, schema_overrides: dict[str, pxt.ColumnType], primary_key: list[str]
) -> tuple[dict[str, pxt.ColumnType], list[str]]:
    """
    Infers a Pixeltable schema from a Pandas DataFrame.

    Returns:
        A tuple containing a Pixeltable schema and a list of primary key column names.
    """
    for pd_name in schema_overrides:
        if pd_name not in df.columns:
            raise excs.Error(
                f'Column `{pd_name}` specified in `schema_overrides` does not exist in the given `DataFrame`.'
            )
    for pd_name in primary_key:
        if pd_name not in df.columns:
            raise excs.Error(f'Primary key column `{pd_name}` does not exist in the given `DataFrame`.')

    schema: dict[str, pxt.ColumnType] = {}
    col_mapping: dict[str, str] = {}  # Maps Pandas column names to Pixeltable column names

    for pd_name, pd_dtype in zip(df.columns, df.dtypes):
        if pd_name in schema_overrides:
            pxt_type = schema_overrides[pd_name]
        else:
            # This complicated-looking condition is necessary because we cannot safely call `pd.isna()` on
            # general objects, so we need to check for nulls in the specific cases where we might expect them.
            # isinstance(val, float) will check for NaN values in float columns *as well as* floats appearing
            # in object columns (where Pandas uses NaN as a general null).
            # np.issubdtype(pd_dtype, np.datetime64) checks for NaT values specifically in datetime columns.
            has_na = any(
                (isinstance(val, float) or np.issubdtype(pd_dtype, np.datetime64)) and pd.isna(val)
                for val in df[pd_name]
            )
            if has_na and pd_name in primary_key:
                raise excs.Error(f'Primary key column `{pd_name}` cannot contain null values.')
            pxt_type = __np_dtype_to_pxt_type(pd_dtype, df[pd_name], pd_name not in primary_key)
        pxt_name = __normalize_pxt_col_name(pd_name)
        # Ensure that column names are unique by appending a distinguishing suffix
        # to any collisions
        if pxt_name in schema:
            n = 2
            while f'{pxt_name}_{n}' in schema:
                n += 1
            pxt_name = f'{pxt_name}_{n}'
        schema[pxt_name] = pxt_type
        col_mapping[pd_name] = pxt_name

    pxt_pk = [col_mapping[pk] for pk in primary_key]
    return schema, pxt_pk


def __normalize_pxt_col_name(pd_name: str) -> str:
    """
    Normalizes an arbitrary DataFrame column name into a valid Pixeltable identifier by:
    - replacing any non-ascii or non-alphanumeric characters with an underscore _
    - prefixing the result with the letter 'c' if it starts with an underscore or a number
    """
    id = ''.join(ch if ch.isascii() and ch.isalnum() else '_' for ch in pd_name)
    if id[0].isnumeric():
        id = f'c_{id}'
    elif id[0] == '_':
        id = f'c{id}'
    assert pxt.catalog.is_valid_identifier(id), id
    return id


def __np_dtype_to_pxt_type(np_dtype: np.dtype, data_col: pd.Series, nullable: bool) -> pxt.ColumnType:
    """
    Infers a Pixeltable type based on a Numpy dtype.
    """
    if np.issubdtype(np_dtype, np.integer):
        return pxt.IntType(nullable=nullable)

    if np.issubdtype(np_dtype, np.floating):
        return pxt.FloatType(nullable=nullable)

    if np.issubdtype(np_dtype, np.bool_):
        return pxt.BoolType(nullable=nullable)

    if np.issubdtype(np_dtype, np.character):
        return pxt.StringType(nullable=nullable)

    if np.issubdtype(np_dtype, np.datetime64):
        return pxt.TimestampType(nullable=nullable)

    if np_dtype == np.object_:
        # The `object_` dtype can mean all sorts of things; see if we can infer the Pixeltable type
        # based on the actual data in `data_col`.
        # First drop any null values (they don't contribute to type inference).
        data_col = data_col.dropna()

        if len(data_col) == 0:
            # No non-null values; default to FloatType (the Pandas type of an all-NaN column)
            return pxt.FloatType(nullable=nullable)

        inferred_type = pxt.ColumnType.infer_common_literal_type(data_col)
        if inferred_type is not None:
            return inferred_type.copy(nullable=nullable)

    raise excs.Error(f'Could not infer Pixeltable type of column: {data_col.name} (dtype: {np_dtype})')


def __df_row_to_pxt_row(row: tuple[Any, ...], schema: dict[str, pxt.ColumnType]) -> dict[str, Any]:
    rows = {}
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
        rows[col_name] = val
    return rows

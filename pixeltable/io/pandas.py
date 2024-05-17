from typing import Optional, Any, Iterable

import numpy as np
import pandas as pd

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts


def import_pandas(
    tbl_name: str, df: pd.DataFrame, *, schema_overrides: Optional[dict[str, pxt.ColumnType]] = None
) -> pxt.catalog.InsertableTable:
    """Creates a new `Table` from a Pandas `DataFrame`, with the specified name. The schema of the table
    will be inferred from the `DataFrame`, unless `schema` is specified.

    The column names of the new `Table` will be identical to those in the `DataFrame`, as long as they are valid
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
    """
    schema = _df_to_pxt_schema(df, schema_overrides)
    tbl_rows = (dict(_df_row_to_pxt_row(row, schema)) for row in df.itertuples())
    table = pxt.create_table(tbl_name, schema)
    table.insert(tbl_rows)
    return table


def import_csv(
    table_path: str, filepath_or_buffer, schema_overrides: Optional[dict[str, ts.ColumnType]] = None, **kwargs
) -> pxt.catalog.InsertableTable:
    """
    Creates a new `Table` from a csv file. This is a convenience method and is equivalent
    to calling `import_pandas(table_path, pd.read_csv(filepath_or_buffer, **kwargs), schema=schema)`.
    See the Pandas documentation for `read_csv` for more details.
    """
    df = pd.read_csv(filepath_or_buffer, **kwargs)
    return import_pandas(table_path, df, schema_overrides=schema_overrides)


def import_excel(
    table_path: str, io, *args, schema_overrides: Optional[dict[str, ts.ColumnType]] = None, **kwargs
) -> pxt.catalog.InsertableTable:
    """
    Creates a new `Table` from an excel (.xlsx) file. This is a convenience method and is equivalent
    to calling `import_pandas(table_path, pd.read_excel(io, *args, **kwargs), schema=schema)`.
    See the Pandas documentation for `read_excel` for more details.
    """
    df = pd.read_excel(io, *args, **kwargs)
    return import_pandas(table_path, df, schema_overrides=schema_overrides)


def _df_to_pxt_schema(
    df: pd.DataFrame, schema_overrides: Optional[dict[str, pxt.ColumnType]]
) -> dict[str, pxt.ColumnType]:
    if schema_overrides is not None:
        for pd_name in schema_overrides:
            if pd_name not in df.columns:
                raise excs.Error(
                    f'Column `{pd_name}` specified in `schema_overrides` does not exist in the given `DataFrame`.'
                )
    schema = {}
    for pd_name, pd_dtype in zip(df.columns, df.dtypes):
        if schema_overrides is not None and pd_name in schema_overrides:
            pxt_type = schema_overrides[pd_name]
        else:
            pxt_type = _np_dtype_to_pxt_type(pd_dtype, df[pd_name])
        pxt_name = _normalize_pxt_col_name(pd_name)
        # Ensure that column names are unique by appending a distinguishing suffix
        # to any collisions
        if pxt_name in schema:
            n = 2
            while f'{pxt_name}_{n}' in schema:
                n += 1
            pxt_name = f'{pxt_name}_{n}'
        schema[pxt_name] = pxt_type
    return schema


def _normalize_pxt_col_name(pd_name: str) -> str:
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


def _np_dtype_to_pxt_type(np_dtype: np.dtype, data_col: pd.Series) -> pxt.ColumnType:
    """
    Infers a Pixeltable type based on a Numpy dtype.
    """
    if np.issubdtype(np_dtype, np.integer):
        return pxt.IntType()
    if np.issubdtype(np_dtype, np.floating):
        return pxt.FloatType()
    if np.issubdtype(np_dtype, np.bool_):
        return pxt.BoolType()
    if np_dtype == np.object_ or np.issubdtype(np_dtype, np.character):
        has_nan = any(isinstance(val, float) and np.isnan(val) for val in data_col)
        return pxt.StringType(nullable=has_nan)
    if np.issubdtype(np_dtype, np.datetime64):
        has_nat = any(pd.isnull(val) for val in data_col)
        return pxt.TimestampType(nullable=has_nat)
    raise excs.Error(f'Unsupported dtype: {np_dtype}')


def _df_row_to_pxt_row(row: tuple[Any, ...], schema: dict[str, pxt.ColumnType]) -> dict[str, Any]:
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

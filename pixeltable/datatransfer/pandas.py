from typing import Optional, Any, Iterable

import numpy as np
import pandas as pd

import pixeltable as pxt
import pixeltable.exceptions as excs


def import_pandas(
        cl: pxt.Client,
        tbl_name: str,
        df: pd.DataFrame,
        *,
        schema: Optional[dict[str, pxt.ColumnType]] = None
) -> pxt.catalog.InsertableTable:
    if schema is None:
        # Infer schema
        schema = _df_to_pxt_schema(df)
    tbl_rows = (
        dict(_df_row_to_pxt_row(row, schema))
        for row in df.itertuples()
    )
    table = cl.create_table(tbl_name, schema)
    table.insert(tbl_rows)
    return table


def _df_to_pxt_schema(df: pd.DataFrame) -> dict[str, pxt.ColumnType]:
    cols = [
        (_normalize_pxt_col_name(pd_name), _np_dtype_to_pxt_type(pd_dtype))
        for pd_name, pd_dtype in zip(df.columns, df.dtypes)
    ]
    schema = {}
    for col_name, col_type in cols:
        # Ensure that column names are unique by appending a unique suffix
        # to any collisions
        if col_name in schema:
            n = 2
            while f'{col_name}_{n}' in schema:
                n += 1
            col_name = f'{col_name}_{n}'
        schema[col_name] = col_type
    return schema


def _normalize_pxt_col_name(pd_name: str) -> str:
    id = ''.join(
        ch if ch.isascii() and ch.isalnum() else '_'
        for ch in pd_name
    )
    if id[0].isnumeric():
        id = f'c_{id}'
    elif id[0] == '_':
        id = f'c{id}'
    assert pxt.catalog.is_valid_identifier(id), id
    return id


def _np_dtype_to_pxt_type(np_dtype: np.dtype) -> pxt.ColumnType:
    if np.issubdtype(np_dtype, np.integer):
        return pxt.IntType()
    if np.issubdtype(np_dtype, np.floating):
        return pxt.FloatType()
    if np.issubdtype(np_dtype, np.bool_):
        return pxt.BoolType()
    if np_dtype == np.object_ or np.issubdtype(np_dtype, np.character):
        return pxt.StringType()
    raise excs.Error(f'Unsupported dtype: {np_dtype}')


def _df_row_to_pxt_row(row: tuple[Any, ...], schema: dict[str, pxt.ColumnType]) -> Iterable[tuple[str, Any]]:
    for val, (col_name, pxt_type) in zip(row[1:], schema.items()):
        if pxt_type.is_int_type():
            val = int(val)
        if pxt_type.is_float_type():
            val = float(val)
        if pxt_type.is_bool_type():
            val = bool(val)
        if pxt_type.is_string_type():
            val = str(val)
        yield col_name, val
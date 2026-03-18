from __future__ import annotations

import csv
import json
import logging
import os
import typing
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas._typing import DtypeObj  # For pandas dtype type hints
from pandas.api.types import is_datetime64_any_dtype, is_extension_array_dtype

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.env import Env
from pixeltable.runtime import get_runtime

if typing.TYPE_CHECKING:
    import pixeltable as pxt

_logger = logging.getLogger('pixeltable')


def import_pandas(
    tbl_name: str,
    df: pd.DataFrame,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
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
    filepath_or_buffer: str | os.PathLike,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
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
    io: str | os.PathLike,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
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

    raise excs.Error(f'Could not infer Pixeltable type of column: {data_col.name} (dtype: {pd_dtype})')


def _df_row_to_pxt_row(
    row: tuple[Any, ...], schema: dict[str, ts.ColumnType], col_mapping: dict[str, str] | None
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
        elif pxt_type.is_uuid_type():
            if pd.isnull(val):
                nval = None
            elif isinstance(val, uuid.UUID):
                nval = val
            else:
                nval = uuid.UUID(val)
        else:
            nval = val
        pxt_row[pxt_name] = nval
    return pxt_row


def export_csv(
    table_or_query: pxt.Table | pxt.Query,
    file_path: str | Path,
    *,
    delimiter: str = ',',
    quoting: int = csv.QUOTE_MINIMAL,
) -> None:
    """
    Exports a query result or table to a CSV file.

    Pixeltable column types are mapped to CSV values as follows:

    - String, Int, Float, Bool: native CSV representation
    - Timestamp, Date: ISO 8601 string representation
    - UUID: string representation
    - Json: JSON-encoded string
    - Array: JSON-encoded string (via ``tolist()``)
    - Binary: excluded from export (not representable in CSV)
    - Image, Video, Audio, Document: file path or URL string

    Args:
        table_or_query: Table or Query to export.
        file_path: Path to the output CSV file.
        delimiter: Field delimiter character. Default ``','``.
        quoting: CSV quoting style (a ``csv.QUOTE_*`` constant). Default ``csv.QUOTE_MINIMAL``.
    """
    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    # Build export list (skip binary columns)
    col_names: list[str] = []
    col_types: list[ts.ColumnType] = []
    for col_name, col_type in query.schema.items():
        if col_type.is_binary_type():
            continue
        col_names.append(col_name)
        col_types.append(col_type)

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    schema_keys = list(query.schema.keys())

    try:
        with get_runtime().catalog.begin_xact(for_write=False), open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=delimiter, quoting=quoting)  # type: ignore[arg-type]
            writer.writerow(col_names)

            select_exprs = query._select_list_exprs
            for data_row in query._exec():
                csv_row: list[Any] = []
                for col_name, col_type in zip(col_names, col_types):
                    idx = schema_keys.index(col_name)
                    slot_idx = select_exprs[idx].slot_idx
                    if col_type.is_image_type():
                        # Images are loaded as PIL.Image; export the file path instead
                        val = data_row.file_paths[slot_idx] or data_row.file_urls[slot_idx]
                    elif col_type.is_media_type():
                        # Video, Audio, Document values are already path strings
                        val = data_row[slot_idx]
                    else:
                        val = data_row[slot_idx]
                    if val is None:
                        csv_row.append('')
                    elif col_type.is_json_type():
                        csv_row.append(json.dumps(val))
                    elif col_type.is_array_type():
                        csv_row.append(json.dumps(val.tolist() if isinstance(val, np.ndarray) else val))
                    else:
                        csv_row.append(val)
                writer.writerow(csv_row)

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)

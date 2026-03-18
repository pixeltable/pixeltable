from __future__ import annotations

import datetime
import json
import logging
import typing
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.runtime import get_runtime

if typing.TYPE_CHECKING:
    import pixeltable as pxt

_logger = logging.getLogger('pixeltable')


def _infer_schema_from_rows(
    rows: Iterable[dict[str, Any]], schema_overrides: dict[str, ts.ColumnType], primary_key: list[str]
) -> dict[str, ts.ColumnType]:
    schema: dict[str, ts.ColumnType] = {}
    cols_with_nones: set[str] = set()

    for n, row in enumerate(rows):
        for col_name, value in row.items():
            if col_name in schema_overrides:
                # We do the insertion here; this will ensure that the column order matches the order
                # in which the column names are encountered in the input data, even if `schema_overrides`
                # is specified.
                if col_name not in schema:
                    assert isinstance(schema_overrides[col_name], ts.ColumnType)
                    schema[col_name] = schema_overrides[col_name]
            elif value is not None:
                # If `key` is not in `schema_overrides`, then we infer its type from the data.
                # The column type will always be nullable by default.
                col_type = ts.ColumnType.infer_literal_type(value, nullable=col_name not in primary_key)
                if col_type is None:
                    raise excs.Error(
                        f'Could not infer type for column `{col_name}`; the value in row {n} '
                        f'has an unsupported type: {type(value)}'
                    )
                if col_name not in schema:
                    schema[col_name] = col_type
                else:
                    supertype = schema[col_name].supertype(col_type, for_inference=True)
                    if supertype is None:
                        raise excs.Error(
                            f'Could not infer type of column `{col_name}`; the value in row {n} '
                            f'does not match preceding type {schema[col_name]}: {value!r}\n'
                            'Consider specifying the type explicitly in `schema_overrides`.'
                        )
                    schema[col_name] = supertype
            else:
                cols_with_nones.add(col_name)

    entirely_none_cols = cols_with_nones - schema.keys()
    if len(entirely_none_cols) > 0:
        # A column can only end up in `entirely_none_cols` if it was not in `schema_overrides` and
        # was not encountered in any row with a non-None value.
        raise excs.Error(
            f'The following columns have no non-null values: {", ".join(entirely_none_cols)}\n'
            'Consider specifying the type(s) explicitly in `schema_overrides`.'
        )
    return schema


def import_rows(
    tbl_path: str,
    rows: list[dict[str, Any]],
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
) -> pxt.Table:
    """
    Creates a new base table from a list of dictionaries. The dictionaries must be of the
    form `{column_name: value, ...}`. Pixeltable will attempt to infer the schema of the table from the
    supplied data, using the most specific type that can represent all the values in a column.

    If `schema_overrides` is specified, then for each entry `(column_name, type)` in `schema_overrides`,
    Pixeltable will force the specified column to the specified type (and will not attempt any type inference
    for that column).

    All column types of the new table will be nullable unless explicitly specified as non-nullable in
    `schema_overrides`.

    Args:
        tbl_path: The qualified name of the table to create.
        rows: The list of dictionaries to import.
        schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            as described above.
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        num_retained_versions: The number of retained versions of the table
            (see [`create_table()`][pixeltable.create_table]).
        comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        tbl_path,
        source=rows,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
    )


def import_json(
    tbl_path: str,
    filepath_or_url: str,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    num_retained_versions: int = 10,
    comment: str = '',
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from a JSON file. This is a convenience method and is
    equivalent to calling `import_data(table_path, json.loads(file_contents, **kwargs), ...)`, where `file_contents`
    is the contents of the specified `filepath_or_url`.

    Args:
        tbl_path: The name of the table to create.
        filepath_or_url: The path or URL of the JSON file.
        schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            (see [`import_rows()`][pixeltable.io.import_rows]).
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        num_retained_versions: The number of retained versions of the table
            (see [`create_table()`][pixeltable.create_table]).
        comment: A comment to attach to the table (see [`create_table()`][pixeltable.create_table]).
        kwargs: Additional keyword arguments to pass to `json.loads`.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    return pxt.create_table(
        tbl_path,
        source=filepath_or_url,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        num_retained_versions=num_retained_versions,
        comment=comment,
        extra_args=kwargs,
    )


def export_json(table_or_query: pxt.Table | pxt.Query, file_path: str | Path, *, indent: int | None = None) -> None:
    """
    Exports a query result or table to a JSON file.

    The output is a JSON array of objects, where each object represents a row. Pixeltable column types
    are mapped to JSON values as follows:

    - String: string
    - Int: number
    - Float: number
    - Bool: boolean
    - Timestamp: ISO 8601 string
    - Date: ISO 8601 string
    - UUID: string
    - Json: native JSON value (object, array, etc.)
    - Array: nested JSON array (via ``tolist()``)
    - Binary: excluded from export (not representable in JSON)
    - Image, Video, Audio, Document: file path or URL string

    Args:
        table_or_query: Table or Query to export.
        file_path: Path to the output JSON file.
        indent: Number of spaces for pretty-printing indentation. Default ``None`` (compact output).
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
    rows: list[dict[str, Any]] = []

    try:
        with get_runtime().catalog.begin_xact(for_write=False):
            select_exprs = query._select_list_exprs
            for data_row in query._exec():
                row_dict: dict[str, Any] = {}
                for col_name, col_type in zip(col_names, col_types):
                    idx = schema_keys.index(col_name)
                    slot_idx = select_exprs[idx].slot_idx
                    if col_type.is_image_type():
                        val = data_row.file_paths[slot_idx] or data_row.file_urls[slot_idx]
                    elif col_type.is_media_type():
                        val = data_row[slot_idx]
                    else:
                        val = data_row[slot_idx]
                    if val is None:
                        row_dict[col_name] = None
                    elif col_type.is_timestamp_type():
                        row_dict[col_name] = val.isoformat() if isinstance(val, datetime.datetime) else str(val)
                    elif col_type.is_date_type():
                        row_dict[col_name] = val.isoformat() if isinstance(val, datetime.date) else str(val)
                    elif col_type.is_uuid_type():
                        row_dict[col_name] = str(val) if isinstance(val, uuid.UUID) else val
                    elif col_type.is_array_type():
                        row_dict[col_name] = val.tolist() if isinstance(val, np.ndarray) else val
                    else:
                        row_dict[col_name] = val
                rows.append(row_dict)

    except excs.ExprEvalError as e:
        query._raise_expr_eval_err(e)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=indent, default=str)

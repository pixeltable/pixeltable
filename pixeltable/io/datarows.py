from __future__ import annotations

from typing import Any, Iterable, Optional, Union

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import exceptions as excs


def _infer_schema_from_rows(
    rows: Iterable[dict[str, Any]], schema_overrides: dict[str, Any], primary_key: list[str]
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
                    supertype = schema[col_name].supertype(col_type)
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
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
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
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
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

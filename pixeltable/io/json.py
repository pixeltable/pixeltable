from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pixeltable as pxt
import pixeltable.type_system as ts

if typing.TYPE_CHECKING:
    import pixeltable as pxt


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

    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    col_types: dict[str, ts.ColumnType] = {name: ct for name, ct in query.schema.items() if not ct.is_binary_type()}

    result = query.collect()

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for row in result:
        row_dict: dict[str, Any] = {}
        for col_name, col_type in col_types.items():
            val = row[col_name]

            if val is None:
                row_dict[col_name] = None
            elif col_type.is_image_type():
                row_dict[col_name] = str(val.filename) if hasattr(val, 'filename') and val.filename else None
            elif col_type.is_timestamp_type() or col_type.is_date_type():
                row_dict[col_name] = val.isoformat()
            elif col_type.is_uuid_type():
                row_dict[col_name] = str(val)
            elif col_type.is_array_type():
                row_dict[col_name] = val.tolist()
            else:
                row_dict[col_name] = val
        rows.append(row_dict)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=indent, default=str, ensure_ascii=False, allow_nan=False)

from __future__ import annotations

import json
import typing
from pathlib import Path
from typing import Any

import pixeltable as pxt
from pixeltable.io.utils import atomic_write, replace_media_with_fileurl

if typing.TYPE_CHECKING:
    import pixeltable as pxt


def import_json(
    tbl_path: str,
    filepath_or_url: str,
    *,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    comment: str | None = None,
    **kwargs: Any,
) -> pxt.Table:
    """
    Creates a new base table from a JSON file. This is a convenience method and is
    equivalent to calling [`create_table()`][pixeltable.create_table] with
    `pxt.create_table(tbl_path, source=filepath_or_url, extra_args=kwargs, ...)`.
    The contents of `filepath_or_url` are read and parsed as JSON internally (using `json.loads(**kwargs)`).

    Args:
        tbl_path: The name of the table to create.
        filepath_or_url: The path or URL of the JSON file.
        schema_overrides: If specified, then columns in `schema_overrides` will be given the specified types
            (see [`import_rows()`][pixeltable.io.import_rows]).
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
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
        comment=comment,
        extra_args=kwargs,
    )


def export_json(table_or_query: pxt.Table | pxt.Query, file_path: str | Path) -> None:
    """
    Exports a query result or table to a JSONL file.

    Pixeltable column types are mapped to JSON values as follows:

    - String: string
    - Int: number
    - Float: number
    - Bool: boolean
    - Timestamp: ISO 8601 string
    - Date: ISO 8601 string
    - UUID: string
    - Json: native JSON value (object, array, etc.)
    - Array: nested JSON array (via `tolist()`)
    - Binary: excluded from export (not representable in JSON)
    - Image, Video, Audio, Document: file path or URL string

    Args:
        table_or_query: Table or Query to export.
        file_path: Path to the output JSONL file.
    """

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    query = query._replace_select_list(replace_media_with_fileurl(query._select_list_exprs))

    cursor = query.cursor()

    file_path = Path(file_path)

    with atomic_write(file_path, mode='w', encoding='utf-8') as f:
        for row in cursor:
            json.dump(row.to_json(), f, ensure_ascii=False)
            f.write('\n')

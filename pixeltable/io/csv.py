from __future__ import annotations

import csv
import json
import os
import typing
from pathlib import Path
from typing import Any

import pixeltable as pxt
import pixeltable.type_system as ts

if typing.TYPE_CHECKING:
    import pixeltable as pxt


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
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    col_types: dict[str, ts.ColumnType] = {name: ct for name, ct in query.schema.items() if not ct.is_binary_type()}

    result = query.collect()

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=delimiter, quoting=quoting)  # type: ignore[arg-type]
        writer.writerow(col_types.keys())

        for row in result:
            csv_row: list[Any] = []
            for col_name, col_type in col_types.items():
                val = row[col_name]
                if val is None:
                    csv_row.append('')
                elif col_type.is_image_type():
                    csv_row.append(str(val.filename) if hasattr(val, 'filename') and val.filename else '')
                elif col_type.is_timestamp_type() or col_type.is_date_type():
                    csv_row.append(val.isoformat())
                elif col_type.is_json_type():
                    csv_row.append(json.dumps(val))
                elif col_type.is_array_type():
                    csv_row.append(json.dumps(val.tolist()))
                else:
                    csv_row.append(val)
            writer.writerow(csv_row)

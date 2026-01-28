from __future__ import annotations

import json
import logging
import typing
from pathlib import Path
from typing import Any

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Catalog
from pixeltable.utils.transactional_directory import transactional_directory

if typing.TYPE_CHECKING:
    import pixeltable as pxt

_logger = logging.getLogger('pixeltable')


def export_parquet(
    table_or_query: pxt.Table | pxt.Query,
    parquet_path: Path,
    partition_size_bytes: int = 100_000_000,
    inline_images: bool = False,
) -> None:
    """
    Exports a query result or table to one or more Parquet files. Requires pyarrow to be installed.

    It additionally writes the pixeltable metadata in a json file, which would otherwise
    not be available in the parquet format.

    Pixeltable column types are mapped to Parquet types as follows:

    - String: string
    - Int: int64
    - Float: float32
    - Bool: bool
    - Timestamp: timestamp[us, tz=UTC]
    - Date: date32
    - UUID: uuid
    - Binary: binary
    - Image: binary (when `inline_images=True`)
    - Audio, Video, Document: string (file paths)
    - Array: fixed_shape_tensor
    - Json: struct

        - Schema is inferred from data via `pyarrow.infer_type()`
        - Fields that contain empty dicts cannot be mapped to a Parquet type and will result in an exception

    Args:
        table_or_query : Table or Query to export.
        parquet_path : Path to directory to write the parquet files to.
        partition_size_bytes : The maximum target size for each chunk. Default 100_000_000 bytes.
        inline_images : If True, images are stored inline in the parquet file. This is useful
                        for small images, to be imported as pytorch dataset. But can be inefficient
                        for large images, and cannot be imported into pixeltable.
                        If False, will raise an error if the Query has any image column.
                        Default False.
    """
    import pyarrow as pa

    from pixeltable.utils.arrow import to_record_batches

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    if not inline_images and any(col_type.is_image_type() for col_type in query.schema.values()):
        raise excs.Error('Cannot export Query with image columns when inline_images is False')

    # store the changes atomically
    with transactional_directory(parquet_path) as temp_path:
        # dump metadata json file so we can inspect what was the source of the parquet file later on.
        json.dump(query.as_dict(), (temp_path / '.pixeltable.json').open('w'))
        type_dict = {k: v.as_dict() for k, v in query.schema.items()}
        json.dump(type_dict, (temp_path / '.pixeltable.column_types.json').open('w'))  # keep type metadata
        batch_num = 0
        with Catalog.get().begin_xact(for_write=False):
            for record_batch in to_record_batches(query, partition_size_bytes):
                output_path = temp_path / f'part-{batch_num:05d}.parquet'
                arrow_tbl = pa.Table.from_batches([record_batch])
                pa.parquet.write_table(arrow_tbl, str(output_path))
                batch_num += 1


def import_parquet(
    table: str,
    *,
    parquet_path: str,
    schema_overrides: dict[str, Any] | None = None,
    primary_key: str | list[str] | None = None,
    **kwargs: Any,
) -> pxt.Table:
    """Creates a new base table from a Parquet file or set of files. Requires pyarrow to be installed.

    Args:
        table: Fully qualified name of the table to import the data into.
        parquet_path: Path to an individual Parquet file or directory of Parquet files.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
            name `name` will be given type `type`, instead of being inferred from the Parquet dataset. The keys in
            `schema_overrides` should be the column names of the Parquet dataset (whether or not they are valid
            Pixeltable identifiers).
        primary_key: The primary key of the table (see [`create_table()`][pixeltable.create_table]).
        kwargs: Additional arguments to pass to `create_table`.

    Returns:
        A handle to the newly created table.
    """
    value = kwargs.pop('source_format', None)
    return pxt.create_table(
        table,
        source=parquet_path,
        source_format=value,
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        extra_args=kwargs,
    )

from __future__ import annotations

import json
import os
import typing
from pathlib import Path
from typing import Any, Literal, cast

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable.io.export import null_image_error
from pixeltable.utils.transactional_directory import transactional_directory

if typing.TYPE_CHECKING:
    import pixeltable as pxt


def export_parquet(
    table_or_query: pxt.Table | pxt.Query,
    parquet_path: str | os.PathLike,
    partition_size_bytes: int = 100_000_000,
    inline_images: bool = False,
    if_exists: Literal['error', 'replace'] = 'error',
    _write_md: bool = False,
    _reject_null_images: bool = False,
) -> None:
    """
    Exports a query result or table to one or more Parquet files. Requires pyarrow to be installed.

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
    - Array (requires shape to be known):
        - fixed_shape_tensor for fixed-shape arrays
        - list for ragged arrays (one or more dimensions are None)
    - Json: struct

        - Schema is inferred from data via `pyarrow.infer_type()`
        - Fields that contain empty dicts cannot be mapped to a Parquet type and will result in an exception

    Args:
        table_or_query : Table or Query to export.
        parquet_path : Path to directory to write the parquet files to. Must not be a symbolic link.
        partition_size_bytes : The maximum target size for each chunk. Default 100_000_000 bytes.
        inline_images : If True, images are stored inline in the parquet file. This is useful
                        for small images, to be imported as pytorch dataset. But can be inefficient
                        for large images, and cannot be imported into pixeltable.
                        If False, will raise an error if the Query has any image column.
                        Default False.
        if_exists: Determines the behavior if `parquet_path` already exists. Must be one of the following:

            - `'error'`: raise an error
            - `'replace'`: delete the existing directory and all of its contents and write the new export in its
              place. The existing directory is removed only after the new export has been written successfully.
              Raises an error if `parquet_path` is a file rather than a directory.
    """
    import pyarrow as pa

    from pixeltable.utils.arrow import to_record_batches

    if if_exists not in ('error', 'replace'):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_ARGUMENT, "export_parquet(): `if_exists` must be one of: ['error', 'replace']"
        )

    path = Path(parquet_path)
    if path.is_symlink():
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f"export_parquet(): '{path!s}' is a symbolic link; pass the directory it points to instead",
        )
    if if_exists == 'replace' and path.exists() and not path.is_dir():
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f"export_parquet(): '{path!s}' is a file, not a directory; `if_exists='replace'` only replaces directories",
        )

    query: pxt.Query
    if isinstance(table_or_query, pxt.catalog.Table):
        query = table_or_query.select()
    else:
        query = table_or_query

    # KNOWN RACE: schema is snapshotted here, outside any xact; to_record_batches below
    # accesses query.schema again under its own xact. For SELECT * queries, a concurrent
    # schema mutation between the two reads can drift the validation here from what
    # to_record_batches sees.
    schema = query.schema
    for col_name, col_type in schema.items():
        if col_type.is_image_type() and not inline_images:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'Cannot export image column {col_name!r} with `inline_images=False`',
            )
        if col_type.is_array_type() and cast(ts.ArrayType, col_type).shape is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Cannot export array column {col_name!r} with unknown shape'
            )

    # store the changes atomically
    with transactional_directory(path, replace=(if_exists == 'replace')) as temp_path:
        if _write_md:
            json.dump(query.as_dict(), (temp_path / '.pixeltable.json').open('w'))
            type_dict = {k: v.as_dict() for k, v in schema.items()}
            json.dump(type_dict, (temp_path / '.pixeltable.column_types.json').open('w'))  # keep type metadata

        image_exprs = (
            {name: expr for expr, name in query._effective_select_list if expr.col_type.is_image_type()}
            if _reject_null_images
            else {}
        )
        for batch_num, record_batch in enumerate(to_record_batches(query, partition_size_bytes)):
            for name, expr in image_exprs.items():
                if record_batch.column(name).null_count > 0:
                    raise null_image_error(expr)
            output_path = temp_path / f'part-{batch_num:05d}.parquet'
            arrow_tbl = pa.Table.from_batches([record_batch])
            pa.parquet.write_table(arrow_tbl, str(output_path))


def import_parquet(
    table: str,
    *,
    parquet_path: str | os.PathLike,
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
    return pxt.create_table(
        table,
        source=parquet_path,
        source_format=kwargs.pop('source_format', 'parquet'),
        schema_overrides=schema_overrides,
        primary_key=primary_key,
        extra_args=kwargs,
    )

from __future__ import annotations

import datetime
import io
import json
import logging
import typing
from collections import deque
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.catalog import Catalog
from pixeltable.utils.transactional_directory import transactional_directory

if typing.TYPE_CHECKING:
    import pyarrow as pa

    import pixeltable as pxt

_logger = logging.getLogger('pixeltable')


def _write_batch(value_batch: dict[str, deque], schema: pa.Schema, output_path: Path) -> None:
    import pyarrow as pa
    from pyarrow import parquet

    pydict = {}
    for field in schema:
        if isinstance(field.type, pa.FixedShapeTensorType):
            stacked_arr = np.stack(value_batch[field.name])
            pydict[field.name] = pa.FixedShapeTensorArray.from_numpy_ndarray(stacked_arr)
        else:
            pydict[field.name] = value_batch[field.name]

    tab = pa.Table.from_pydict(pydict, schema=schema)
    parquet.write_table(tab, str(output_path))


def export_parquet(
    table_or_df: Union[pxt.Table, pxt.DataFrame],
    parquet_path: Path,
    partition_size_bytes: int = 100_000_000,
    inline_images: bool = False,
) -> None:
    """
    Exports a dataframe's data to one or more Parquet files. Requires pyarrow to be installed.

    It additionally writes the pixeltable metadata in a json file, which would otherwise
    not be available in the parquet format.

    Args:
        table_or_df : Table or Dataframe to export.
        parquet_path : Path to directory to write the parquet files to.
        partition_size_bytes : The maximum target size for each chunk. Default 100_000_000 bytes.
        inline_images : If True, images are stored inline in the parquet file. This is useful
                        for small images, to be imported as pytorch dataset. But can be inefficient
                        for large images, and cannot be imported into pixeltable.
                        If False, will raise an error if the Dataframe has any image column.
                        Default False.
    """
    from pixeltable.utils.arrow import to_arrow_schema

    df: pxt.DataFrame
    if isinstance(table_or_df, pxt.catalog.Table):
        df = table_or_df._df()
    else:
        df = table_or_df

    type_dict = {k: v.as_dict() for k, v in df.schema.items()}
    arrow_schema = to_arrow_schema(df.schema)

    if not inline_images and any(col_type.is_image_type() for col_type in df.schema.values()):
        raise excs.Error('Cannot export Dataframe with image columns when inline_images is False')

    # store the changes atomically
    with transactional_directory(parquet_path) as temp_path:
        # dump metadata json file so we can inspect what was the source of the parquet file later on.
        json.dump(df.as_dict(), (temp_path / '.pixeltable.json').open('w'))
        json.dump(type_dict, (temp_path / '.pixeltable.column_types.json').open('w'))  # keep type metadata

        batch_num = 0
        current_value_batch: dict[str, deque] = {k: deque() for k in df.schema}
        current_byte_estimate = 0

        with Catalog.get().begin_xact(for_write=False):
            for data_row in df._exec():
                for (col_name, col_type), e in zip(df.schema.items(), df._select_list_exprs):
                    val = data_row[e.slot_idx]
                    if val is None:
                        current_value_batch[col_name].append(val)
                        continue

                    assert val is not None
                    if col_type.is_image_type():
                        # images get inlined into the parquet file
                        if data_row.file_paths is not None and data_row.file_paths[e.slot_idx] is not None:
                            # if there is a file, read directly to preserve information
                            with open(data_row.file_paths[e.slot_idx], 'rb') as f:
                                val = f.read()
                        elif isinstance(val, PIL.Image.Image):
                            # if no file available, eg. bc it is computed, convert to png
                            buf = io.BytesIO()
                            val.save(buf, format='PNG')
                            val = buf.getvalue()
                        else:
                            raise excs.Error(f'unknown image type {type(val)}')
                        length = len(val)
                    elif col_type.is_string_type():
                        length = len(val)
                    elif col_type.is_video_type() or col_type.is_audio_type():
                        if data_row.file_paths is not None and data_row.file_paths[e.slot_idx] is not None:
                            val = data_row.file_paths[e.slot_idx]
                        else:
                            raise excs.Error(f'unknown audio/video type {type(val)}')
                        length = len(val)
                    elif col_type.is_json_type():
                        val = json.dumps(val)
                        length = len(val)
                    elif col_type.is_array_type():
                        length = val.nbytes
                    elif col_type.is_int_type() or col_type.is_float_type():
                        length = 8
                    elif col_type.is_bool_type():
                        length = 1
                    elif col_type.is_date_type():
                        length = 4
                    elif col_type.is_timestamp_type():
                        val = val.astimezone(datetime.timezone.utc)
                        length = 8
                    else:
                        raise excs.Error(f'unknown type {col_type} for {col_name}')

                    current_value_batch[col_name].append(val)
                    current_byte_estimate += length
                if current_byte_estimate > partition_size_bytes:
                    assert batch_num < 100_000, 'wrote too many parquet files, unclear ordering'
                    _write_batch(current_value_batch, arrow_schema, temp_path / f'part-{batch_num:05d}.parquet')
                    batch_num += 1
                    current_value_batch = {k: deque() for k in df.schema}
                    current_byte_estimate = 0

            _write_batch(current_value_batch, arrow_schema, temp_path / f'part-{batch_num:05d}.parquet')


def import_parquet(
    table: str,
    *,
    parquet_path: str,
    schema_overrides: Optional[dict[str, Any]] = None,
    primary_key: Optional[Union[str, list[str]]] = None,
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

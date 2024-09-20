from __future__ import annotations

import io
import json
import logging
import random
import typing
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Any

import PIL.Image
import numpy as np

import pixeltable.exceptions as exc
import pixeltable.type_system as ts
from pixeltable.utils.transactional_directory import transactional_directory

if typing.TYPE_CHECKING:
    import pixeltable as pxt
    import pyarrow as pa
    from pyarrow import parquet

_logger = logging.getLogger(__name__)


def _write_batch(value_batch: Dict[str, deque], schema: pa.Schema, output_path: Path) -> None:
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
    parquet.write_table(tab, output_path)


def save_parquet(df: pxt.DataFrame, dest_path: Path, partition_size_bytes: int = 100_000_000) -> None:
    """
    Internal method to stream dataframe data to parquet format.
    Does not materialize the dataset to memory.

    It preserves pixeltable type metadata in a json file, which would otherwise
    not be available in the parquet format.

    Images are stored inline in a compressed format in their parquet file.

    Args:
        df : dataframe to save.
        dest_path : path to directory to save the parquet files to.
        partition_size_bytes : maximum target size for each chunk. Default 100_000_000 bytes.
    """
    from pixeltable.utils.arrow import to_arrow_schema

    type_dict = {k: v.as_dict() for k, v in df.schema.items()}
    arrow_schema = to_arrow_schema(df.schema)

    # store the changes atomically
    with transactional_directory(dest_path) as temp_path:
        # dump metadata json file so we can inspect what was the source of the parquet file later on.
        json.dump(df.as_dict(), (temp_path / '.pixeltable.json').open('w'))
        json.dump(type_dict, (temp_path / '.pixeltable.column_types.json').open('w'))  # keep type metadata

        batch_num = 0
        current_value_batch: Dict[str, deque] = {k: deque() for k in df.schema.keys()}
        current_byte_estimate = 0

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
                        assert False, f'unknown image type {type(val)}'
                    length = len(val)
                elif col_type.is_string_type():
                    length = len(val)
                elif col_type.is_video_type():
                    if data_row.file_paths is not None and data_row.file_paths[e.slot_idx] is not None:
                        val = data_row.file_paths[e.slot_idx]
                    else:
                        assert False, f'unknown video type {type(val)}'
                    length = len(val)
                elif col_type.is_json_type():
                    val = json.dumps(val)
                    length = len(val)
                elif col_type.is_array_type():
                    length = val.nbytes
                elif col_type.is_int_type():
                    length = 8
                elif col_type.is_float_type():
                    length = 8
                elif col_type.is_bool_type():
                    length = 1
                elif col_type.is_timestamp_type():
                    length = 8
                else:
                    assert False, f'unknown type {col_type} for {col_name}'

                current_value_batch[col_name].append(val)
                current_byte_estimate += length
            if current_byte_estimate > partition_size_bytes:
                assert batch_num < 100_000, 'wrote too many parquet files, unclear ordering'
                _write_batch(current_value_batch, arrow_schema, temp_path / f'part-{batch_num:05d}.parquet')
                batch_num += 1
                current_value_batch = {k: deque() for k in df.schema.keys()}
                current_byte_estimate = 0

        _write_batch(current_value_batch, arrow_schema, temp_path / f'part-{batch_num:05d}.parquet')


def parquet_schema_to_pixeltable_schema(parquet_path: str) -> Dict[str, Optional[ts.ColumnType]]:
    """Generate a default pixeltable schema for the given parquet file. Returns None for unknown types."""
    from pyarrow import parquet
    from pixeltable.utils.arrow import to_pixeltable_schema

    input_path = Path(parquet_path).expanduser()
    parquet_dataset = parquet.ParquetDataset(input_path)
    return to_pixeltable_schema(parquet_dataset.schema)


def import_parquet(
    table_path: str,
    *,
    parquet_path: str,
    schema_overrides: Optional[Dict[str, ts.ColumnType]] = None,
    **kwargs: Any,
) -> pxt.Table:
    """Creates a new base table from a Parquet file or set of files. Requires pyarrow to be installed.

    Args:
        table_path: Path to the table.
        parquet_path: Path to an individual Parquet file or directory of Parquet files.
        schema_overrides: If specified, then for each (name, type) pair in `schema_overrides`, the column with
            name `name` will be given type `type`, instead of being inferred from the `DataFrame`. The keys in
            `schema_overrides` should be the column names of the `DataFrame` (whether or not they are valid
            Pixeltable identifiers).
        kwargs: Additional arguments to pass to `create_table`.

    Returns:
        A handle to the newly created [`Table`][pixeltable.Table].
    """
    import pixeltable as pxt
    from pyarrow import parquet
    from pixeltable.utils.arrow import iter_tuples

    input_path = Path(parquet_path).expanduser()
    parquet_dataset = parquet.ParquetDataset(input_path)

    schema = parquet_schema_to_pixeltable_schema(parquet_path)
    if schema_overrides is None:
        schema_overrides = {}

    schema.update(schema_overrides)
    for k, v in schema.items():
        if v is None:
            raise exc.Error(f'Could not infer pixeltable type for column {k} from parquet file')

    if table_path in pxt.list_tables():
        raise exc.Error(f'Table {table_path} already exists')

    try:
        tmp_name = f'{table_path}_tmp_{random.randint(0, 100000000)}'
        tab = pxt.create_table(tmp_name, schema, **kwargs)
        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches():
                dict_batch = list(iter_tuples(batch))
                tab.insert(dict_batch)
    except Exception as e:
        _logger.error(f'Error while inserting Parquet file into table: {e}')
        raise e

    pxt.move(tmp_name, table_path)
    return pxt.get_table(table_path)

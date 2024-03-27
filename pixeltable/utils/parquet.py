import io
import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import PIL.Image
import pyarrow as pa
import pyarrow.parquet

import pixeltable.type_system as ts
from pixeltable.utils.arrow import iter_tuples, to_arrow_schema, to_pixeltable_schema
from pixeltable.utils.transactional_directory import transactional_directory
import pixeltable.exceptions as exc

import random

_logger = logging.getLogger(__name__)


def _write_batch(value_batch : Dict[str, deque], schema : pa.Schema, output_path : Path) -> None:
    pydict = {}
    for field in schema:
        if isinstance(field.type, pa.FixedShapeTensorType):
            stacked_arr = np.stack(value_batch[field.name])
            pydict[field.name] = pa.FixedShapeTensorArray.from_numpy_ndarray(stacked_arr)
        else:
            pydict[field.name] = value_batch[field.name]

    tab = pa.Table.from_pydict(pydict, schema=schema)
    pa.parquet.write_table(tab, output_path)

def save_parquet(df: 'pixeltable.DataFrame', dest_path: Path, partition_size_bytes: int = 100_000_000) -> None:
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
    column_names = df.get_column_names()
    column_types = df.get_column_types()
    type_dict = {k: v.as_dict() for k, v in zip(column_names, column_types)}
    arrow_schema = to_arrow_schema(dict(zip(column_names, column_types)))

    # store the changes atomically
    with transactional_directory(dest_path) as temp_path:
        # dump metadata json file so we can inspect what was the source of the parquet file later on.
        json.dump(df._as_dict(), (temp_path / '.pixeltable.json').open('w')) # pylint: disable=protected-access
        json.dump(type_dict, (temp_path / '.pixeltable.column_types.json').open('w')) # keep type metadata

        batch_num = 0
        current_value_batch : Dict[str, deque] = {k:deque() for k in column_names}
        current_byte_estimate = 0

        for data_row in df._exec(): # pylint: disable=protected-access
            for (col_name, col_type, e) in zip(column_names, column_types, df._select_list_exprs): # pylint: disable=protected-access
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
                _write_batch(current_value_batch, arrow_schema,  temp_path / f'part-{batch_num:05d}.parquet')
                batch_num += 1
                current_value_batch = {k:deque() for k in column_names}
                current_byte_estimate = 0

        _write_batch(current_value_batch, arrow_schema, temp_path / f'part-{batch_num:05d}.parquet')


def parquet_schema_to_pixeltable_schema(parquet_path: str) -> Dict[str, Optional[ts.ColumnType]]:
    """Generate a default pixeltable schema for the given parquet file. Returns None for unknown types."""

    input_path = Path(parquet_path).expanduser()
    parquet_dataset = pa.parquet.ParquetDataset(input_path)
    return to_pixeltable_schema(parquet_dataset.schema)


def import_parquet(
    cl: 'pixeltable.Client',
    table_path: str,
    *,
    parquet_path: str,
    schema_override: Optional[Dict[str, ts.ColumnType]],
    **kwargs,
) -> 'catalog.InsertableTable':
    """See `pixeltable.Client.import_parquet` for documentation"""
    input_path = Path(parquet_path).expanduser()
    parquet_dataset = pa.parquet.ParquetDataset(input_path)

    schema = parquet_schema_to_pixeltable_schema(parquet_path)
    if schema_override is None:
        schema_override = {}

    schema.update(schema_override)
    for k, v in schema.items():
        if v is None:
            raise exc.Error(f'Could not infer pixeltable type for column {k} from parquet file')

    if table_path in cl.list_tables():
        raise exc.Error(f'Table {table_path} already exists')

    try:
        tmp_name = f'{table_path}_tmp_{random.randint(0, 100000000)}'
        tab = cl.create_table(tmp_name, schema, **kwargs)
        for fragment in parquet_dataset.fragments:
            for batch in fragment.to_batches():
                dict_batch = list(iter_tuples(batch))
                tab.insert(dict_batch)
    except Exception as e:
        _logger.error(f'Error while inserting Parquet file into table: {e}')
        raise e

    cl.move(tmp_name, table_path)
    return cl.get_table(table_path)

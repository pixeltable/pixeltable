import json
import PIL.Image
import io
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from pathlib import Path
from collections import deque

from typing import List, Tuple, Any, Dict
from pixeltable.utils.transactional_directory import transactional_directory

def _write_batch(value_batch : Dict[str, deque], schema : pa.Schema, output_path : Path) -> None:
    pydict = {}
    for field in schema:
        if isinstance(field.type, pa.FixedShapeTensorType):
            stacked_arr = np.stack(value_batch[field.name])
            pydict[field.name] = pa.FixedShapeTensorArray.from_numpy_ndarray(stacked_arr)
        else:
            pydict[field.name] = value_batch[field.name]

    tab = pa.Table.from_pydict(pydict, schema=schema)
    pq.write_table(tab, output_path)

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

    arrow_schema = pa.schema([pa.field(name, column_types[i].to_arrow_type()) for i, name in enumerate(column_names)])
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
                        val = open(data_row.file_paths[e.slot_idx], 'rb').read()
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

def get_part_metadata(path : Path) -> List[Tuple[str, int]]:
    """
        Args:
            path: path to directory containing parquet files.
        Returns:
            A list of (file_name, num_rows) tuples for the parquet files in file name order.
    """
    parts = sorted([f for f in path.iterdir() if f.suffix == '.parquet'])
    rows_per_file = {}

    for part in parts:
        parquet_file = pq.ParquetFile(str(part))
        rows_per_file[part] = parquet_file.metadata.num_rows

    return [(file, num_rows) for file, num_rows in rows_per_file.items()]
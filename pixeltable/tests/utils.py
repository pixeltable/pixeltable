import datetime
import glob
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import PIL

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ColumnType, StringType, IntType, FloatType, BoolType, TimestampType


def make_default_type(t: ColumnType.Type) -> ColumnType:
    if t == ColumnType.Type.STRING:
        return StringType()
    if t == ColumnType.Type.INT:
        return IntType()
    if t == ColumnType.Type.FLOAT:
        return FloatType()
    if t == ColumnType.Type.BOOL:
        return BoolType()
    if t == ColumnType.Type.TIMESTAMP:
        return TimestampType()
    assert False

def make_tbl(cl: pt.Client, name: str = 'test', col_names: List[str] = ['c1']) -> catalog.MutableTable:
    schema: List[catalog.Column] = []
    for i, col_name in enumerate(col_names):
        schema.append(catalog.Column(f'{col_name}', make_default_type(ColumnType.Type(i % 5))))
    return cl.create_table(name, schema)

def create_table_data(t: catalog.Table, col_names: List[str] = [], num_rows: int = 10) -> List[List[Any]]:
    data: Dict[str, Any] = {}
    sample_dict = {
        'detections': [{
            'id': '637e8e073b28441a453564cf',
            'attributes': {},
            'tags': [],
            'label': 'potted plant',
            'bounding_box': [
                0.37028125,
                0.3345305164319249,
                0.038593749999999996,
                0.16314553990610328,
            ],
            'mask': None,
            'confidence': None,
            'index': None,
            'supercategory': 'furniture',
            'iscrowd': 0,
        }, {
            'id': '637e8e073b28441a453564cf',
            'attributes': {},
            'tags': [],
            'label': 'potted plant',
            'bounding_box': [
                0.37028125,
                0.3345305164319249,
                0.038593749999999996,
                0.16314553990610328,
            ],
            'mask': None,
            'confidence': None,
            'index': None,
            'supercategory': 'furniture',
            'iscrowd': 0,
        }]
    }

    if len(col_names) == 0:
        col_names = [c.name for c in t.columns if not c.is_computed]

    for col_name in col_names:
        col = t.cols_by_name[col_name]
        col_data: Any = None
        if col.col_type.is_string_type():
            col_data = ['test string'] * num_rows
        if col.col_type.is_int_type():
            col_data = np.random.randint(0, 100, size=num_rows).tolist()
        if col.col_type.is_float_type():
            col_data = (np.random.random(size=num_rows) * 100).tolist()
        if col.col_type.is_bool_type():
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]
        if col.col_type.is_timestamp_type():
            col_data = [datetime.datetime.now()] * num_rows
        if col.col_type.is_json_type():
            col_data = [sample_dict] * num_rows
        if col.col_type.is_array_type():
            col_data = [np.ones(col.col_type.shape, dtype=col.col_type.numpy_dtype()) for i in range(num_rows)]
        if col.col_type.is_image_type():
            image_path = get_image_files()[0]
            col_data = [image_path for i in range(num_rows)]
        if col.col_type.is_video_type():
            video_path = get_video_files()[0]
            col_data = [video_path for i in range(num_rows)]
        data[col.name] = col_data
    rows = [[data[col_name][i] for col_name in col_names] for i in range(num_rows)]
    return rows

def read_data_file(dir_name: str, file_name: str, path_col_names: List[str] = []) -> Tuple[List[List[Any]], List[str]]:
    """
    Locate dir_name, create df out of file_name.
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
    Returns:
        tuple of (list of rows, list of column names)
    """
    glob_result = glob.glob(f'{os.getcwd()}/**/{dir_name}', recursive=True)
    assert len(glob_result) == 1, f'Could not find {dir_name}'
    abs_path = Path(glob_result[0])
    data_file_path = abs_path / file_name
    assert data_file_path.is_file(), f'Not a file: {str(data_file_path)}'
    df = pd.read_csv(str(data_file_path))
    for col_name in path_col_names:
        assert col_name in df.columns
        df[col_name] = df.apply(lambda r: str(abs_path / r[col_name]), axis=1)
    return df.values.tolist(), df.columns.tolist()

def get_video_files() -> List[str]:
    glob_result = glob.glob(f'{os.getcwd()}/**/videos/*', recursive=True)
    return glob_result

def get_image_files() -> List[str]:
    glob_result = glob.glob(f'{os.getcwd()}/**/imagenette2-160/*', recursive=True)
    return glob_result
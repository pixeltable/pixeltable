import datetime
import glob
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ColumnType, StringType, IntType, FloatType, BoolType, TimestampType
from pixeltable.function import Function

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

def make_tbl(db: pt.Db, name: str = 'test', col_names: List[str] = ['c1']) -> pt.MutableTable:
    schema: List[catalog.Column] = []
    for i, col_name in enumerate(col_names):
        schema.append(catalog.Column(f'{col_name}', make_default_type(ColumnType.Type(i % 5))))
    return db.create_table(name, schema)

def create_table_data(t: catalog.Table, col_names: List[str] = [], num_rows: int = 10) -> pd.DataFrame:
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
        col_names = [c.name for c in t.columns]

    for col_name in col_names:
        col = t.cols_by_name[col_name]
        col_data: Any = None
        if col.col_type.is_string_type():
            col_data = ['test string'] * num_rows
        if col.col_type.is_int_type():
            col_data = np.random.randint(0, 100, size=num_rows)
        if col.col_type.is_float_type():
            col_data = np.random.random(size=num_rows) * 100
        if col.col_type.is_bool_type():
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]
        if col.col_type.is_timestamp_type():
            col_data = datetime.datetime.now()
        if col.col_type.is_json_type():
            col_data = [sample_dict] * num_rows
        # TODO: implement this
        assert not col.col_type.is_image_type()
        assert not col.col_type.is_array_type()
        data[col.name] = col_data
    return pd.DataFrame(data=data)

def read_data_file(dir_name: str, file_name: str, path_col_names: List[str] = []) -> pd.DataFrame:
    """
    Locate dir_name, create df out of file_name.
    transform columns 'file_name' to column 'file_path' with absolute paths
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
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
        return df

def get_video_files() -> List[str]:
    glob_result = glob.glob(f'{os.getcwd()}/**/videos/*.mp4', recursive=True)
    return glob_result


class SumAggregator:
    def __init__(self):
        self.sum = 0
    @classmethod
    def make_aggregator(cls) -> 'SumAggregator':
        return cls()
    def update(self, val: int) -> None:
        self.sum += val
    def value(self) -> int:
        return self.sum

sum_uda = Function(
    IntType(), [IntType()],
    init_fn=SumAggregator.make_aggregator, update_fn=SumAggregator.update, value_fn=SumAggregator.value)

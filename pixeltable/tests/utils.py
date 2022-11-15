import datetime
import glob
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ColumnType


def make_tbl(db: pt.Db, name: str = 'test', col_names: List[str] = ['c1']) -> pt.MutableTable:
    schema: List[catalog.Column] = []
    for i, col_name in enumerate(col_names):
        schema.append(catalog.Column(f'{col_name}', ColumnType(i % len(ColumnType))))
    return db.create_table(name, schema)

def create_table_data(t: catalog.Table, num_rows: int = 10) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for col in t.columns():
        col_data: Any = None
        if col.col_type == ColumnType.STRING:
            col_data = ['test string'] * num_rows
        if col.col_type == ColumnType.INT:
            col_data = np.random.randint(0, 100, size=num_rows)
        if col.col_type == ColumnType.FLOAT:
            col_data = np.random.random(size=num_rows) * 100
        if col.col_type == ColumnType.BOOL:
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]
        if col.col_type == ColumnType.TIMESTAMP:
            col_data = datetime.datetime.now()
        # TODO: implement this
        assert col.col_type != ColumnType.IMAGE
        assert col.col_type != ColumnType.DICT
        assert col.col_type != ColumnType.VECTOR
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

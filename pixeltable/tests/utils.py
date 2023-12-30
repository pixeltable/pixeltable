import datetime
import glob
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import \
    ColumnType, StringType, IntType, FloatType, ArrayType, BoolType, TimestampType, JsonType, ImageType, VideoType
from pixeltable.dataframe import DataFrameResultSet


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

def make_tbl(cl: pt.Client, name: str = 'test', col_names: List[str] = ['c1']) -> catalog.InsertableTable:
    schema: List[catalog.Column] = []
    for i, col_name in enumerate(col_names):
        schema.append(catalog.Column(f'{col_name}', make_default_type(ColumnType.Type(i % 5))))
    return cl.create_table(name, schema)

def create_table_data(t: catalog.MutableTable, col_names: List[str] = [], num_rows: int = 10) -> List[Dict[str, Any]]:
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
        col_names = [c.name for c in t.columns() if not c.is_computed]

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
    rows = [{col_name: data[col_name][i] for col_name in col_names} for i in range(num_rows)]
    return rows

def create_test_tbl(client: pt.Client) -> catalog.MutableTable:
    cols = [
        catalog.Column('c1', StringType(nullable=False)),
        catalog.Column('c1n', StringType(nullable=True)),
        catalog.Column('c2', IntType(nullable=False), primary_key=True),
        catalog.Column('c3', FloatType(nullable=False)),
        catalog.Column('c4', BoolType(nullable=False)),
        catalog.Column('c5', TimestampType(nullable=False)),
        catalog.Column('c6', JsonType(nullable=False)),
        catalog.Column('c7', JsonType(nullable=False)),
    ]
    t = client.create_table('test_tbl', cols)
    t.add_column(catalog.Column('c8', computed_with=[[1, 2, 3], [4, 5, 6]]))

    num_rows = 100
    d1 = {
        'f1': 'test string 1',
        'f2': 1,
        'f3': 1.0,
        'f4': True,
        'f5': [1.0, 2.0, 3.0, 4.0],
        'f6': {
            'f7': 'test string 2',
            'f8': [1.0, 2.0, 3.0, 4.0],
        },
    }
    d2 = [d1, d1]

    c1_data = [f'test string {i}' for i in range(num_rows)]
    c2_data = [i for i in range(num_rows)]
    c3_data = [float(i) for i in range(num_rows)]
    c4_data = [bool(i % 2) for i in range(num_rows)]
    c5_data = [datetime.datetime.now()] * num_rows
    c6_data = []
    for i in range(num_rows):
        d = {
            'f1': f'test string {i}',
            'f2': i,
            'f3': float(i),
            'f4': bool(i % 2),
            'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {
                'f7': 'test string 2',
                'f8': [1.0, 2.0, 3.0, 4.0],
            },
        }
        c6_data.append(d)

    c7_data = [d2] * num_rows
    rows = [
        {
            'c1': c1_data[i],
            'c1n': c1_data[i] if i % 10 != 0 else None,
            'c2': c2_data[i],
            'c3': c3_data[i],
            'c4': c4_data[i],
            'c5': c5_data[i],
            'c6': c6_data[i],
            'c7': c7_data[i],
        }
        for i in range(num_rows)
    ]
    t.insert(rows)
    return t

def create_all_datatype_tbl(test_client: pt.Client) -> catalog.Table:
    """ Creates a table with all supported datatypes.
    """
    cols = [
        catalog.Column('row_id', IntType(nullable=False)), # used for row selection
        catalog.Column('c_array', ArrayType(shape=(10,),  dtype=FloatType(), nullable=True)),
        catalog.Column('c_bool', BoolType(nullable=True)),
        catalog.Column('c_float', FloatType(nullable=True)),
        catalog.Column('c_image', ImageType(nullable=True)),
        catalog.Column('c_int', IntType(nullable=True)),
        catalog.Column('c_json', JsonType(nullable=True)),
        catalog.Column('c_string', StringType(nullable=True)),
        catalog.Column('c_timestamp', TimestampType(nullable=True)),
        catalog.Column('c_video', VideoType(nullable=True)),
    ]
    tbl = test_client.create_table('all_datatype_tbl', cols)
    example_rows = create_table_data(tbl, num_rows=11)

    for i,r in enumerate(example_rows):
        r['row_id'] = i # row_id

    tbl.insert(example_rows)
    return tbl

def read_data_file(dir_name: str, file_name: str, path_col_names: List[str] = []) -> List[Dict[str, Any]]:
    """
    Locate dir_name, create df out of file_name.
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
    Returns:
        tuple of (list of rows, list of column names)
    """
    tests_dir = os.path.dirname(__file__) # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/{dir_name}', recursive=True)
    assert len(glob_result) == 1, f'Could not find {dir_name}'
    abs_path = Path(glob_result[0])
    data_file_path = abs_path / file_name
    assert data_file_path.is_file(), f'Not a file: {str(data_file_path)}'
    df = pd.read_csv(str(data_file_path))
    for col_name in path_col_names:
        assert col_name in df.columns
        df[col_name] = df.apply(lambda r: str(abs_path / r[col_name]), axis=1)
    return df.to_dict(orient='records')

def get_video_files(include_bad_video=False) -> List[str]:
    tests_dir = os.path.dirname(__file__) # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/videos/*', recursive=True)
    if not include_bad_video:
        glob_result = [f for f in glob_result if 'bad_video' not in f]
    return glob_result

def get_image_files() -> List[str]:
    tests_dir = os.path.dirname(__file__) # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/imagenette2-160/*', recursive=True)
    return glob_result

def get_audio_files(include_bad_audio=False) -> List[str]:
    tests_dir = os.path.dirname(__file__)
    glob_result = glob.glob(f'{tests_dir}/**/audio/*', recursive=True)
    if not include_bad_audio:
        glob_result = [f for f in glob_result if 'bad_audio' not in f]
    return glob_result

def assert_resultset_eq(r1: DataFrameResultSet, r2: DataFrameResultSet) -> None:
    assert len(r1) == len(r2)
    assert len(r1.column_names()) == len(r2.column_names())  # we don't care about the actual column names
    r1_pd = r1.to_pandas()
    r2_pd = r2.to_pandas()
    for i in range(len(r1.column_names())):
        # only compare column values
        s1 = r1_pd.iloc[:, i]
        s2 = r2_pd.iloc[:, i]
        if s1.dtype == np.float64:
            assert np.allclose(s1, s2)
        else:
            assert s1.equals(s2)
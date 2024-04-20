import datetime
import glob
import json
import os
from collections import namedtuple
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import PIL.Image
import numpy as np
import pandas as pd
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import catalog
from pixeltable.catalog.globals import UpdateStatus
from pixeltable.dataframe import DataFrameResultSet
from pixeltable.env import Env
from pixeltable.functions.huggingface import clip_image, clip_text
from pixeltable.type_system import (
    ArrayType,
    BoolType,
    ColumnType,
    FloatType,
    ImageType,
    IntType,
    JsonType,
    StringType,
    TimestampType,
    VideoType,
)


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


def make_tbl(cl: pxt.Client, name: str = 'test', col_names: Optional[List[str]] = None) -> catalog.InsertableTable:
    if col_names is None:
        col_names = ['c1']
    schema: Dict[str, ts.ColumnType] = {}
    for i, col_name in enumerate(col_names):
        schema[f'{col_name}'] = make_default_type(ColumnType.Type(i % 5))
    return cl.create_table(name, schema)


def create_table_data(t: catalog.Table, col_names: Optional[List[str]] = None, num_rows: int = 10) -> List[
    Dict[str, Any]]:
    if col_names is None:
        col_names = []
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

    col_types = t.column_types()
    for col_name in col_names:
        col_type = col_types[col_name]
        col_data: Any = None
        if col_type.is_string_type():
            col_data = ['test string'] * num_rows
        if col_type.is_int_type():
            col_data = np.random.randint(0, 100, size=num_rows).tolist()
        if col_type.is_float_type():
            col_data = (np.random.random(size=num_rows) * 100).tolist()
        if col_type.is_bool_type():
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]
        if col_type.is_timestamp_type():
            col_data = [datetime.datetime.now()] * num_rows
        if col_type.is_json_type():
            col_data = [sample_dict] * num_rows
        if col_type.is_array_type():
            col_data = [np.ones(col_type.shape, dtype=col_type.numpy_dtype()) for i in range(num_rows)]
        if col_type.is_image_type():
            image_path = get_image_files()[0]
            col_data = [image_path for i in range(num_rows)]
        if col_type.is_video_type():
            video_path = get_video_files()[0]
            col_data = [video_path for i in range(num_rows)]
        data[col_name] = col_data
    rows = [{col_name: data[col_name][i] for col_name in col_names} for i in range(num_rows)]
    return rows


def create_test_tbl(client: pxt.Client, name: str = 'test_tbl') -> catalog.Table:
    schema = {
        'c1': StringType(nullable=False),
        'c1n': StringType(nullable=True),
        'c2': IntType(nullable=False),
        'c3': FloatType(nullable=False),
        'c4': BoolType(nullable=False),
        'c5': TimestampType(nullable=False),
        'c6': JsonType(nullable=False),
        'c7': JsonType(nullable=False),
    }
    t = client.create_table(name, schema, primary_key='c2')
    t.add_column(c8=[[1, 2, 3], [4, 5, 6]])

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


def create_img_tbl(cl: pxt.Client, name: str = 'test_img_tbl') -> catalog.Table:
    schema = {
        'img': ImageType(nullable=False),
        'category': StringType(nullable=False),
        'split': StringType(nullable=False),
    }
    tbl = cl.create_table(name, schema)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    tbl.insert(rows)
    return tbl


def create_all_datatypes_tbl(test_client: pxt.Client) -> catalog.Table:
    """ Creates a table with all supported datatypes.
    """
    schema = {
        'row_id': IntType(nullable=False),  # used for row selection
        'c_array': ArrayType(shape=(10,), dtype=FloatType(), nullable=True),
        'c_bool': BoolType(nullable=True),
        'c_float': FloatType(nullable=True),
        'c_image': ImageType(nullable=True),
        'c_int': IntType(nullable=True),
        'c_json': JsonType(nullable=True),
        'c_string': StringType(nullable=True),
        'c_timestamp': TimestampType(nullable=True),
        'c_video': VideoType(nullable=True),
    }
    tbl = test_client.create_table('all_datatype_tbl', schema)
    example_rows = create_table_data(tbl, num_rows=11)

    for i, r in enumerate(example_rows):
        r['row_id'] = i  # row_id

    tbl.insert(example_rows)
    return tbl


def read_data_file(dir_name: str, file_name: str, path_col_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Locate dir_name, create df out of file_name.
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
    Returns:
        tuple of (list of rows, list of column names)
    """
    if path_col_names is None:
        path_col_names = []
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
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


def get_video_files(include_bad_video: bool = False) -> List[str]:
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/videos/*', recursive=True)
    if not include_bad_video:
        glob_result = [f for f in glob_result if 'bad_video' not in f]

    half_res = [f for f in glob_result if 'half_res' in f or 'bad_video' in f]
    return half_res


def get_test_video_files() -> List[str]:
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/test_videos/*', recursive=True)
    return glob_result


def get_image_files(include_bad_image: bool = False) -> List[str]:
    tests_dir = os.path.dirname(__file__)  # search with respect to tests/ dir
    glob_result = glob.glob(f'{tests_dir}/**/imagenette2-160/*', recursive=True)
    if not include_bad_image:
        glob_result = [f for f in glob_result if 'bad_image' not in f]
    return glob_result


def get_audio_files(include_bad_audio: bool = False) -> List[str]:
    tests_dir = os.path.dirname(__file__)
    glob_result = glob.glob(f'{tests_dir}/**/audio/*', recursive=True)
    if not include_bad_audio:
        glob_result = [f for f in glob_result if 'bad_audio' not in f]
    return glob_result


def get_documents() -> List[str]:
    tests_dir = os.path.dirname(__file__)
    # for now, we can only handle .html and .md
    return [p for p in glob.glob(f'{tests_dir}/**/documents/*', recursive=True) if not p.endswith('.pdf')]


def get_sentences(n: int = 100) -> List[str]:
    tests_dir = os.path.dirname(__file__)
    path = glob.glob(f'{tests_dir}/**/jeopardy.json', recursive=True)[0]
    with open(path, 'r', encoding='utf8') as f:
        questions_list = json.load(f)
    # this dataset contains \' around the questions
    return [q['question'].replace("'", '') for q in questions_list[:n]]


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


def skip_test_if_not_installed(package) -> None:
    if not Env.get().is_installed_package(package):
        pytest.skip(f'Package `{package}` is not installed.')


def validate_update_status(status: UpdateStatus, expected_rows: Optional[int] = None) -> None:
    assert status.num_excs == 0
    if expected_rows is not None:
        assert status.num_rows == expected_rows


def make_test_arrow_table(output_path: Path) -> None:
    import pyarrow as pa

    value_dict = {
        'c_id': [1, 2, 3, 4, 5],
        'c_int64': [-10, -20, -30, -40, None],
        'c_int32': [-1, -2, -3, -4, None],
        'c_float32': [1.1, 2.2, 3.3, 4.4, None],
        'c_string': ['aaa', 'bbb', 'ccc', 'ddd', None],
        'c_boolean': [True, False, True, False, None],
        'c_timestamp': [
            datetime.datetime(2012, 1, 1, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 2, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 3, 12, 0, 0, 25),
            datetime.datetime(2012, 1, 4, 12, 0, 0, 25),
            None,
        ],
        # The pyarrow fixed_shape_tensor type does not support NULLs (currently can write them but not read them)
        # So, no nulls in this column
        'c_array_float32': [
            [
                1.0,
                2.0,
            ],
            [
                10.0,
                20.0,
            ],
            [
                100.0,
                200.0,
            ],
            [
                1000.0,
                2000.0,
            ],
            [10000.0, 20000.0],
        ],
    }

    arr_size = len(value_dict['c_array_float32'][0])
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (arr_size,))

    schema = pa.schema(
        [
            ('c_id', pa.int32()),
            ('c_int64', pa.int64()),
            ('c_int32', pa.int32()),
            ('c_float32', pa.float32()),
            ('c_string', pa.string()),
            ('c_boolean', pa.bool_()),
            ('c_timestamp', pa.timestamp('us')),
            ('c_array_float32', tensor_type),
        ]
    )

    test_table = pa.Table.from_pydict(value_dict, schema=schema)
    pa.parquet.write_table(test_table, str(output_path / 'test.parquet'))


def assert_hf_dataset_equal(hf_dataset: 'datasets.Dataset', df: pxt.DataFrame, split_column_name: str) -> None:
    import datasets
    assert df.count() == hf_dataset.num_rows
    assert set(df.get_column_names()) == (set(hf_dataset.features.keys()) | {split_column_name})

    # immutable so we can use it as in a set
    DatasetTuple = namedtuple('DatasetTuple', ' '.join(hf_dataset.features.keys()))
    acc_dataset: Set[DatasetTuple] = set()
    for tup in hf_dataset:
        immutable_tup = {}
        for k in tup:
            if isinstance(tup[k], list):
                immutable_tup[k] = tuple(tup[k])
            else:
                immutable_tup[k] = tup[k]

        acc_dataset.add(DatasetTuple(**immutable_tup))

    for tup in df.collect():
        assert tup[split_column_name] in hf_dataset.split._name

        encoded_tup = {}
        for column_name, value in tup.items():
            if column_name == split_column_name:
                continue
            feature_type = hf_dataset.features[column_name]
            if isinstance(feature_type, datasets.ClassLabel):
                assert value in feature_type.names
                # must use the index of the class label as the value to
                # compare with dataset iteration output.
                value = feature_type.encode_example(value)
            elif isinstance(feature_type, datasets.Sequence):
                assert feature_type.feature.dtype == 'float32', 'may need to add more types'
                value = tuple([float(x) for x in value])

            encoded_tup[column_name] = value

        check_tup = DatasetTuple(**encoded_tup)
        assert check_tup in acc_dataset

@pxt.expr_udf
def img_embed(img: PIL.Image.Image) -> np.ndarray:
    return clip_image(img, model_id='openai/clip-vit-base-patch32')

@pxt.expr_udf
def text_embed(txt: str) -> np.ndarray:
    return clip_text(txt, model_id='openai/clip-vit-base-patch32')

SAMPLE_IMAGE_URL = \
    'https://raw.githubusercontent.com/pixeltable/pixeltable/master/docs/source/data/images/000000000009.jpg'


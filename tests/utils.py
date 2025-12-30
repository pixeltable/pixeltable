import datetime
import glob
import itertools
import json
import os
import random
import shutil
import subprocess
import sysconfig
import uuid
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import TestCase
from uuid import uuid4

import more_itertools
import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable._query import ResultSet
from pixeltable.catalog import Catalog
from pixeltable.env import Env
from pixeltable.utils import sha256sum
from pixeltable.utils.console_output import ConsoleMessageFilter, ConsoleOutputHandler
from pixeltable.utils.object_stores import ObjectOps

TESTS_DIR = Path(os.path.dirname(__file__))


def runs_linux_with_gpu() -> bool:
    try:
        import torch

        return sysconfig.get_platform() == 'linux-x86_64' and torch.cuda.is_available()
    except ImportError:
        return False


def make_default_type(t: ts.ColumnType.Type) -> ts.ColumnType:
    if t == ts.ColumnType.Type.STRING:
        return ts.StringType()
    if t == ts.ColumnType.Type.INT:
        return ts.IntType()
    if t == ts.ColumnType.Type.FLOAT:
        return ts.FloatType()
    if t == ts.ColumnType.Type.BOOL:
        return ts.BoolType()
    if t == ts.ColumnType.Type.TIMESTAMP:
        return ts.TimestampType()
    if t == ts.ColumnType.Type.DATE:
        return ts.DateType()
    if t == ts.ColumnType.Type.UUID:
        return ts.UUIDType()
    if t == ts.ColumnType.Type.BINARY:
        return ts.BinaryType()
    raise AssertionError()


def make_tbl(name: str = 'test', col_names: list[str] | None = None) -> pxt.Table:
    if col_names is None:
        col_names = ['c1']
    schema: dict[str, ts.ColumnType] = {}
    for i, col_name in enumerate(col_names):
        schema[f'{col_name}'] = make_default_type(ts.ColumnType.Type(i % 5))
    return pxt.create_table(name, schema)


def create_table_data(t: pxt.Table, col_names: list[str] | None = None, num_rows: int = 10) -> list[dict[str, Any]]:
    if col_names is None:
        col_names = []
    data: dict[str, Any] = {}

    # avoid fields with empty dicts, they cannot be serialized as a struct in Parquet
    sample_dict = {
        'detections': [
            {
                'id': '637e8e073b28441a453564cf',
                'tags': [],
                'label': 'potted plant',
                'bounding_box': [0.37028125, 0.3345305164319249, 0.038593749999999996, 0.16314553990610328],
                'mask': None,
                'confidence': None,
                'index': None,
                'supercategory': 'furniture',
                'iscrowd': 0,
            },
            {
                'id': '637e8e073b28441a453564cf',
                'tags': [],
                'label': 'potted plant',
                'bounding_box': [0.37028125, 0.3345305164319249, 0.038593749999999996, 0.16314553990610328],
                'mask': None,
                'confidence': None,
                'index': None,
                'supercategory': 'furniture',
                'iscrowd': 0,
            },
        ]
    }

    if len(col_names) == 0:
        col_names = [c.name for c in t._tbl_version_path.columns() if not c.is_computed]

    col_types = t._get_schema()
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
            col_data = [False if i == 0 else True for i in col_data]  # noqa: SIM211
        if col_type.is_timestamp_type():
            col_data = [datetime.datetime.now()] * num_rows
        if col_type.is_date_type():
            col_data = [datetime.date.today()] * num_rows
        if col_type.is_uuid_type():
            col_data = [uuid.uuid4() for _ in range(num_rows)]
        if col_type.is_binary_type():
            col_data = [b'1$\x03\xfe'] * num_rows
        if col_type.is_json_type():
            col_data = [sample_dict] * num_rows
        if col_type.is_array_type():
            assert isinstance(col_type, ts.ArrayType)
            col_data = [np.ones(col_type.shape, dtype=col_type.dtype)] * num_rows
        if col_type.is_image_type():
            image_path = get_image_files()[0]
            col_data = [image_path] * num_rows
        if col_type.is_video_type():
            video_path = get_video_files()[0]
            col_data = [video_path] * num_rows
        if col_type.is_audio_type():
            audio_path = get_audio_files()[0]
            col_data = [audio_path] * num_rows
        data[col_name] = col_data
    rows = [{col_name: data[col_name][i] for col_name in col_names} for i in range(num_rows)]
    return rows


def create_test_tbl(name: str = 'test_tbl') -> pxt.Table:
    schema = {
        'c1': pxt.Required[pxt.String],
        'c1n': pxt.String,
        'c2': pxt.Required[pxt.Int],
        'c3': pxt.Required[pxt.Float],
        'c4': pxt.Required[pxt.Bool],
        'c5': pxt.Required[pxt.Timestamp],
        'c6': pxt.Required[pxt.Json],
        'c7': pxt.Required[pxt.Json],
    }
    t = pxt.create_table(name, schema, primary_key='c2')
    t.add_computed_column(c8=pxt.array([[1, 2, 3], [4, 5, 6]]))

    num_rows = 100
    d1 = {
        'f1': 'test string 1',
        'f2': 1,
        'f3': 1.0,
        'f4': True,
        'f5': [1.0, 2.0, 3.0, 4.0],
        'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
    }
    d2 = [d1, d1]

    c1_data = [f'test string {i}' for i in range(num_rows)]
    c2_data = list(range(num_rows))
    c3_data = [float(i) for i in range(num_rows)]
    c4_data = [bool(i % 2) for i in range(num_rows)]
    c5_data = [datetime.datetime(2024, 7, 1) + datetime.timedelta(hours=i) for i in range(num_rows)]
    c6_data = []
    for i in range(num_rows):
        d = {
            'f1': f'test string {i}',
            'f2': i,
            'f3': float(i),
            'f4': bool(i % 2),
            'f5': list(range(5 + i // 10)),
            # 'f5': [1.0, 2.0, 3.0, 4.0],
            'f6': {'f7': 'test string 2', 'f8': [1.0, 2.0, 3.0, 4.0]},
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


def create_img_tbl(name: str = 'test_img_tbl', num_rows: int = 0) -> pxt.Table:
    schema = {'img': pxt.Required[pxt.Image], 'category': pxt.Required[pxt.String], 'split': pxt.Required[pxt.String]}
    tbl = pxt.create_table(name, schema)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    if num_rows > 0:
        # select output_rows randomly in the hope of getting a good sample of the available categories
        rng = np.random.default_rng(17)
        idxs = rng.choice(np.arange(len(rows)), size=num_rows, replace=False)
        rows = [rows[i] for i in idxs]
    tbl.insert(rows)
    return tbl


def create_all_datatypes_tbl() -> pxt.Table:
    """Creates a table with all supported datatypes."""
    schema = {
        'row_id': pxt.Required[pxt.Int],
        'c_array': pxt.Array[(10,), pxt.Float],  # type: ignore[misc]
        'c_audio': pxt.Audio,
        'c_bool': pxt.Bool,
        'c_date': pxt.Date,
        'c_float': pxt.Float,
        'c_image': pxt.Image,
        'c_int': pxt.Int,
        'c_json': pxt.Json,
        'c_string': pxt.String,
        'c_timestamp': pxt.Timestamp,
        'c_uuid': pxt.UUID,
        'c_binary': pxt.Binary,
        'c_video': pxt.Video,
    }
    tbl = pxt.create_table('all_datatype_tbl', schema)
    example_rows = create_table_data(tbl, num_rows=11)

    for i, r in enumerate(example_rows):
        r['row_id'] = i  # row_id

    tbl.insert(example_rows)
    return tbl


def create_scalars_tbl(num_rows: int, seed: int = 0, percent_nulls: int = 10) -> pxt.Table:
    """
    Creates a table with scalar columns, each of which contains randomly generated data.
    """
    assert percent_nulls >= 0 and percent_nulls <= 100
    rng = np.random.default_rng(seed)
    schema = {
        'row_id': ts.IntType(nullable=False),  # used for row selection
        'c_bool': ts.BoolType(nullable=True),
        'c_float': ts.FloatType(nullable=True),
        'c_int': ts.IntType(nullable=True),
        'c_string': ts.StringType(nullable=True),
        'c_timestamp': ts.TimestampType(nullable=True),
    }
    tbl = pxt.create_table('scalars_tbl', schema)

    example_rows: list[dict[str, Any]] = []
    str_chars = 'abcdefghijklmnopqrstuvwxyzab'
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2019, 12, 31)
    delta_days = (end_date - start_date).days
    for i in range(num_rows):
        str_idx = int(rng.integers(0, 26))
        days = int(rng.integers(0, delta_days))
        seconds = int(rng.integers(0, 60 * 60 * 24))
        example_rows.append(
            {
                'row_id': i,
                'c_bool': None if rng.integers(0, 100) < percent_nulls else bool(rng.choice([True, False])),
                'c_float': None if rng.integers(0, 100) < percent_nulls else float(rng.uniform(0, 1)),
                'c_int': None if rng.integers(0, 100) < percent_nulls else int(rng.integers(0, 10)),
                'c_string': None if rng.integers(0, 100) < percent_nulls else str_chars[str_idx : str_idx + 3],
                'c_timestamp': None
                if rng.integers(0, 100) < percent_nulls
                else start_date + datetime.timedelta(days=days, seconds=seconds),
            }
        )
    tbl.insert(example_rows)
    return tbl


def inf_array_iterator(
    shapes: list[tuple[int, ...]], dtypes: list[type[np.integer | np.floating | np.bool_]]
) -> Iterator[np.ndarray]:
    """Generate random arrays of different sizes and dtypes."""
    size_iter = itertools.cycle(shapes)
    dtype_iter = itertools.cycle(dtypes)
    rng = np.random.default_rng(0)
    while True:
        size = next(size_iter)
        dtype = next(dtype_iter)
        if dtype is np.bool_:
            yield rng.integers(0, 2, size=size, dtype=bool)
        elif np.issubdtype(dtype, np.integer):
            yield rng.integers(0, 100, size=size, dtype=dtype)
        else:
            yield rng.random(size=size, dtype=dtype)  # type: ignore[arg-type]


def read_data_file(dir_name: str, file_name: str, path_col_names: list[str] | None = None) -> list[dict[str, Any]]:
    """
    Locate dir_name, create df out of file_name.
    path_col_names: col names in csv file that contain file names; those will be converted to absolute paths
    by adding the path to 'file_name' as a prefix.
    Returns:
        tuple of (list of rows, list of column names)
    """
    if path_col_names is None:
        path_col_names = []
    glob_result = glob.glob(f'{TESTS_DIR}/**/{dir_name}', recursive=True)
    assert len(glob_result) == 1, f'Could not find {dir_name}'
    abs_path = Path(glob_result[0])
    data_file_path = abs_path / file_name
    assert data_file_path.is_file(), f'Not a file: {data_file_path}'
    df = pd.read_csv(str(data_file_path))
    for col_name in path_col_names:
        assert col_name in df.columns
        df[col_name] = df.apply(lambda r: str(abs_path / r[col_name]), axis=1)
    return df.to_dict(orient='records')  # type: ignore[return-value]


def get_video_files(include_bad_video: bool = False, include_vfr: bool = True) -> list[str]:
    glob_result = glob.glob(f'{TESTS_DIR}/**/videos/*', recursive=True)
    if not include_bad_video:
        glob_result = [f for f in glob_result if 'bad_video' not in f]
    if not include_vfr:
        glob_result = [f for f in glob_result if 'vfr' not in f]

    half_res = [f for f in glob_result if 'half_res' in f or 'bad_video' in f]
    half_res.sort()
    return half_res


def get_test_video_files() -> list[str]:
    glob_result = glob.glob(f'{TESTS_DIR}/**/test_videos/*', recursive=True)
    return glob_result


def generate_test_video(
    tmp_path: Path,
    duration: float = 1.0,
    size: str = '640x360',
    fps: int = 25,
    has_audio: bool = True,
    pix_fmt: str = 'yuv420p',
) -> str:
    """Create test video with specific properties using ffmpeg and return its path"""
    cmd = ['ffmpeg', '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size={size}:rate={fps}']
    if has_audio:
        cmd.extend(['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}'])
    if Env.get().default_video_encoder is not None:
        cmd.extend(['-c:v', Env.get().default_video_encoder])
    cmd.extend(['-pix_fmt', pix_fmt])
    if has_audio:
        cmd.extend(['-c:a', 'aac'])
    output_path = tmp_path / f'{uuid4()}.mp4'
    cmd.extend(['-y', str(output_path)])
    subprocess.run(cmd, capture_output=True, check=True)
    return str(output_path)


__IMAGE_FILES: list[str] = []
__IMAGE_FILES_WITH_BAD_IMAGE: list[str] = []


# Gets all image files in the test folder.
# The images will be returned in an order that: (1) is deterministic; (2) ensures that images
# of different modes appear early in the list; and (3) is appropriately randomized subject to
# these constraints.
def get_image_files(include_bad_image: bool = False) -> list[str]:
    global __IMAGE_FILES, __IMAGE_FILES_WITH_BAD_IMAGE  # noqa: PLW0603
    if not __IMAGE_FILES:
        img_files_path = TESTS_DIR / 'data' / 'imagenette2-160'
        glob_result = glob.glob(f'{img_files_path}/*.JPEG')
        glob_result.sort()
        assert len(glob_result) > 1000
        bad_image = next(f for f in glob_result if 'bad_image' in f)
        good_images = [(__image_mode(f), f) for f in glob_result if 'bad_image' not in f]
        # Group images by mode
        modes = sorted({mode for mode, _ in good_images})  # Ensure modes are in a deterministic order
        groups = [[f for mode, f in good_images if mode == mode_group] for mode_group in modes]
        # Sort and randomize the images in each group to ensure that the ordering is both
        # deterministic and not dependent on extrinsic characteristics such as filename
        for group in groups:
            group.sort()
            random.Random(4171780).shuffle(group)
        # Combine the groups in round-robin fashion to ensure that small initial segments
        # contain representatives from each group
        __IMAGE_FILES = list(more_itertools.roundrobin(*groups))
        __IMAGE_FILES_WITH_BAD_IMAGE = [bad_image, *__IMAGE_FILES]
    return __IMAGE_FILES_WITH_BAD_IMAGE if include_bad_image else __IMAGE_FILES


def inf_image_iterator(include_bad_image: bool = False) -> Iterator[PIL.Image.Image]:
    for f in itertools.cycle(get_image_files(include_bad_image)):
        yield PIL.Image.open(f)


def __image_mode(path: str) -> str:
    image = PIL.Image.open(path)
    try:
        return image.mode
    finally:
        image.close()


def get_multimedia_commons_video_uris(n: int = 10) -> list[str]:
    uri = 's3://multimedia-commons/data/videos/mp4/'
    return ObjectOps.list_uris(uri, n_max=n)


def get_audio_files(include_bad_audio: bool = False) -> list[str]:
    audio_dir = TESTS_DIR / 'data' / 'audio'
    glob_result = glob.glob(f'{audio_dir}/*', recursive=True)
    if not include_bad_audio:
        glob_result = [f for f in glob_result if 'bad_audio' not in f]
    return glob_result


def get_audio_file(name: str) -> str | None:
    audio_dir = TESTS_DIR / 'data' / 'audio'
    file_path = audio_dir / name
    glob_result = glob.glob(f'{file_path}', recursive=True)
    return glob_result.pop(0) if len(glob_result) > 0 else None


def get_documents() -> list[str]:
    docs_dir = TESTS_DIR / 'data' / 'documents'
    return glob.glob(f'{docs_dir}/*', recursive=True)


def get_sentences(n: int = 100) -> list[str]:
    path = glob.glob(f'{TESTS_DIR}/**/jeopardy.json', recursive=True)[0]
    with open(path, 'r', encoding='utf8') as f:
        questions_list = json.load(f)
    # this dataset contains \' around the questions
    return [q['question'].replace("'", '') for q in questions_list[:n]]


def assert_resultset_eq(r1: ResultSet, r2: ResultSet, compare_col_names: bool = False) -> None:
    assert len(r1) == len(r2)
    assert len(r1.schema) == len(r2.schema)
    assert all(type1.matches(type2) for type1, type2 in zip(r1.schema.values(), r2.schema.values()))
    if compare_col_names:
        assert r1.schema.keys() == r2.schema.keys()
    for r1_col, r2_col in zip(r1.schema, r2.schema):
        assert_columns_eq(r1_col, r1.schema[r1_col], r1[r1_col], r2[r2_col])


def assert_columns_eq(col_name: str, col_type: ts.ColumnType, c1: list[Any], c2: list[Any]) -> None:
    mismatches = __find_column_mismatches(col_type, c1, c2)
    assert len(mismatches) == 0, __mismatch_err_string(col_name, c1, c2, mismatches)


def __find_column_mismatches(col_type: ts.ColumnType, s1: list[Any], s2: list[Any]) -> list[int]:
    """
    Find the first `limit` mismatches between two lists of values.
    """
    comparer = __COMPARERS.get(col_type._type, __equality_comparer)
    mismatches = []
    for i, (v1, v2) in enumerate(zip(s1, s2)):
        if (v1 is None) != (v2 is None) or (v1 is not None and not comparer(v1, v2)):
            mismatches.append(i)
    return mismatches


def __float_comparer(x: float, y: float) -> bool:
    return bool(np.isclose(x, y, equal_nan=True))


def __array_comparer(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(x, y)


def __file_comparer(x: str, y: str) -> bool:
    return sha256sum(x) == sha256sum(y)


def __equality_comparer(x: Any, y: Any) -> bool:
    return x == y


def __json_comparer(x: Any, y: Any) -> bool:
    if type(x) is not type(y):
        return False
    if isinstance(x, dict):
        return set(x.keys()) == set(y.keys()) and all(__json_comparer(x[k], y[k]) for k in x)
    if isinstance(x, list):
        return len(x) == len(y) and all(__json_comparer(a, b) for a, b in zip(x, y))
    if isinstance(x, float):
        return __float_comparer(x, y)
    if isinstance(x, np.ndarray):
        return __array_comparer(x, y)
    return x == y


__COMPARERS: dict[ts.ColumnType.Type, Callable[[Any, Any], bool]] = {
    ts.ColumnType.Type.FLOAT: __float_comparer,
    ts.ColumnType.Type.ARRAY: __array_comparer,
    ts.ColumnType.Type.JSON: __json_comparer,
    ts.ColumnType.Type.VIDEO: __file_comparer,
    ts.ColumnType.Type.AUDIO: __file_comparer,
    ts.ColumnType.Type.DOCUMENT: __file_comparer,
}


def __mismatch_err_string(col_name: str, s1: list[Any], s2: list[Any], mismatches: list[int]) -> str:
    lines = [f'Column {col_name!r} does not match.']
    for i in mismatches[:5]:
        lines.append(f'Row {i}: {s1[i]} != {s2[i]}')
    if len(mismatches) > 5:
        lines.append(f'(... {len(mismatches) - 5} more mismatches)')
    return '\n'.join(lines)


def assert_table_metadata_eq(expected: dict[str, Any], actual: pxt.TableMetadata) -> None:
    """
    Assert that table metadata (user-facing metadata as returned by `tbl.get_metadata()`) matches the expected dict.
    `version_created` will be checked to be less than 1 minute ago; the other fields will be checked for exact
    equality.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    actual_created_at: datetime.datetime = actual['version_created']
    assert (now - actual_created_at).total_seconds() <= 60

    trimmed_actual = {k: v for k, v in actual.items() if k != 'version_created'}
    tc = TestCase()
    tc.maxDiff = 10_000
    tc.assertDictEqual(expected, trimmed_actual)


def assert_version_metadata_eq(expected: dict[str, Any], actual: pxt.VersionMetadata) -> None:
    """
    Assert that version metadata (user-facing metadata as returned by `tbl.get_versions()`) matches the expected dict.
    `created_at` will be checked to be less than 1 minute ago; the other fields will be checked for exact
    equality.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    actual_created_at: datetime.datetime = actual['created_at']
    assert (now - actual_created_at).total_seconds() <= 60

    trimmed_actual = {k: v for k, v in actual.items() if k != 'created_at'}
    tc = TestCase()
    tc.maxDiff = 10_000
    tc.assertDictEqual(expected, trimmed_actual)


def strip_lines(s: str) -> str:
    lines = s.split('\n')
    return '\n'.join(line.strip() for line in lines)


def skip_test_if_not_installed(*packages: str) -> None:
    for package in packages:
        if not Env.get().is_installed_package(package):
            pytest.skip(f'Package `{package}` is not installed.')


def skip_test_if_not_in_path(*binaries: str) -> None:
    for binary in binaries:
        if shutil.which(binary) is None:
            pytest.skip(f'Binary `{binary}` is not in PATH.')


def skip_test_if_no_client(client_name: str) -> None:
    try:
        _ = Env.get().get_client(client_name)
    except pxt.Error as exc:
        pytest.skip(str(exc))


def skip_test_if_no_pxt_credentials() -> None:
    if not Env.get().pxt_api_key:
        pytest.skip('No Pixeltable API key is configured.')


def skip_test_if_no_aws_credentials() -> None:
    import boto3
    from botocore.exceptions import NoCredentialsError

    try:
        cl = boto3.client('s3')
        cl.list_buckets()
    except NoCredentialsError as exc:
        pytest.skip(str(exc))


def validate_update_status(status: pxt.UpdateStatus, expected_rows: int | None = None) -> None:
    assert status.num_excs == 0
    if expected_rows is not None:
        assert status.num_rows == expected_rows, status


def validate_sync_status(
    status: pxt.UpdateStatus,
    expected_external_rows_created: int | None = None,
    expected_external_rows_updated: int | None = None,
    expected_external_rows_deleted: int | None = None,
    expected_pxt_rows_updated: int | None = None,
    expected_num_excs: int | None = 0,
) -> None:
    if expected_external_rows_created is not None:
        assert status.external_rows_created == expected_external_rows_created, status
    if expected_external_rows_updated is not None:
        assert status.external_rows_updated == expected_external_rows_updated, status
    if expected_external_rows_deleted is not None:
        assert status.external_rows_deleted == expected_external_rows_deleted, status
    if expected_pxt_rows_updated is not None:
        assert status.pxt_rows_updated == expected_pxt_rows_updated, status
    if expected_num_excs is not None:
        assert status.num_excs == expected_num_excs, status


def make_test_arrow_table(output_path: Path) -> str:
    import pyarrow as pa
    from pyarrow import parquet

    float_array = [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0], [1000.0, 2000.0], [10000.0, 20000.0]]
    value_dict: dict[str, list] = {
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
        'c_uuid': [uuid.uuid4().bytes, uuid.uuid4().bytes, uuid.uuid4().bytes, uuid.uuid4().bytes, None],
        'c_binary': [b'abc', b'\x03\x05\xfe', b'ghi', b'jkl', None],
        # The pyarrow fixed_shape_tensor type does not support NULLs (currently can write them but not read them)
        # So, no nulls in this column
        'c_array_float32': float_array,
    }

    arr_size = len(float_array[0])
    tensor_type = pa.fixed_shape_tensor(pa.float32(), (arr_size,))

    fields = [
        ('c_id', pa.int32()),
        ('c_int64', pa.int64()),
        ('c_int32', pa.int32()),
        ('c_float32', pa.float32()),
        ('c_string', pa.string()),
        ('c_boolean', pa.bool_()),
        ('c_timestamp', pa.timestamp('us')),
        ('c_uuid', pa.uuid()),
        ('c_binary', pa.binary()),
        ('c_array_float32', tensor_type),
    ]
    schema = pa.schema(fields)  # type: ignore[arg-type]

    test_table = pa.Table.from_pydict(value_dict, schema=schema)
    table_path = str(output_path / 'test.parquet')
    parquet.write_table(test_table, table_path)
    # read it back and verify the tables match
    read_table = parquet.read_table(table_path)
    assert read_table.num_rows == test_table.num_rows
    assert set(read_table.column_names) == set(test_table.column_names)
    assert read_table.schema.equals(test_table.schema)
    assert read_table.equals(test_table)
    return table_path


def reload_catalog(reload: bool = True) -> None:
    if not reload:
        return
    Catalog.clear()
    pxt.init()


@contextmanager
def capture_console_output() -> Iterator[StringIO]:
    try:
        sio = StringIO()
        handler = ConsoleOutputHandler(stream=sio)
        handler.setLevel(10)
        handler.addFilter(ConsoleMessageFilter())
        Env.get()._logger.addHandler(handler)
        yield sio
    finally:
        Env.get()._logger.removeHandler(handler)
        sio.flush()


# Mock UDF for testing LLM tool invocations
@pxt.udf
def stock_price(ticker: str) -> float:
    """
    Get today's stock price for a given ticker symbol.

    Args:
        ticker - The ticker symbol of the stock to look up.
    """
    if ticker == 'NVDA':
        return 131.17
    elif ticker == 'UAL':
        return 82.88
    else:
        # Return 0.0 instead of None, to distinguish between these two cases: the tool not being called, and the tool
        # being called on a symbol other than NVDA/UAL
        return 0.0


SAMPLE_IMAGE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000009.jpg'


class ReloadTester:
    """Utility to verify that queries return identical results after a catalog reload"""

    query_info: list[tuple[dict[str, Any], ResultSet]]  # list of (query.as_dict(), query.collect())

    def __init__(self) -> None:
        self.query_info = []

    def clear(self) -> None:
        self.query_info = []

    def run_query(self, query: pxt.Query) -> ResultSet:
        query_dict = query.as_dict()
        result_set = query.collect()
        self.query_info.append((query_dict, result_set))
        return result_set

    def run_reload_test(self, clear: bool = True) -> None:
        reload_catalog()
        assert len(self.query_info) > 0, 'No queries in ReloadTester!'
        # enumerate(): the list index is useful for debugging
        for _idx, (query_dict, result_set) in enumerate(self.query_info):
            query = pxt.Query.from_dict(query_dict)
            new_result_set = query.collect()
            try:
                assert_resultset_eq(result_set, new_result_set, compare_col_names=True)
            except Exception as e:
                s = f'Reload test failed for query:\n{query}\n{e}'
                raise RuntimeError(s) from e
        if clear:
            self.clear()


def rerun(**kwargs: Any) -> Callable:
    from .conftest import DO_RERUN

    if 'condition' in kwargs:
        kwargs['condition'] = DO_RERUN and kwargs['condition']
    else:
        kwargs['condition'] = DO_RERUN
    if 'only_rerun' not in kwargs:
        # Set this to an explicit empty list to override the global default in cases where the @rerun decorator is used
        kwargs['only_rerun'] = []
    return pytest.mark.flaky(**kwargs)


# This will be set to True if the tests are running in a CI environment.
IN_CI = bool(os.environ.get('PXTTEST_IN_CI'))

# The OS id (e.g., 'unbuntu-latest') on which the tests are running in CI, or None if not in CI.
CI_OS = os.environ.get('PXTTEST_CI_OS')

"""Test helpers for the pixeltable-rs compatibility harness.

A replacement for upstream tests/utils.py: same names and behavior for every helper the selected test files (and the
harness conftest) use, implemented with Pixeltable's public API only, plus the test hook
`pixeltable._testing.reload_catalog()`. No pixeltable.env / Env / Config / runtime / catalog / exprs / func internals
and no sqlalchemy. Where upstream reaches into internals, the replacement and its difference are noted in the
docstring and in compat/README.md.

Module-level UDF and iterator definitions are guarded: if pixeltable-rs cannot define one yet, the name is bound to an
object that raises `UtilsDefinitionError` when used, so that this module (and with it every test file) still imports.
"""

import asyncio
import datetime
import gc
import glob
import hashlib
import importlib.util
import io
import itertools
import json
import logging
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TypedDict
from unittest import TestCase
from uuid import uuid4

import more_itertools
import numpy as np
import pandas as pd
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs

TESTS_DIR = Path(os.path.dirname(__file__))


DbRootId = Literal['local', 'proxy', 'cloud', 'cloud-cli', 'cloud-serving']

# pixeltable-rs has no hosted databases; kept so that code referring to them still imports.
CLOUD_DB_ROOT_URIS = {'cloud': 'pxt://pixeltable:pxttest', 'cloud-cli': 'pxt://pixeltable:pxttest-cli'}


def new_db_uri() -> str:
    return f'pxt://pixeltable:pxttest-{uuid.uuid4().hex[:12]}'


def cloud_env_configured() -> bool:
    """Always False: the harness runs only against the in-process ('local') database root."""
    return False


@dataclass
class DatabaseRoot:
    id: DbRootId
    base_uri: str
    prefix: str

    @property
    def is_cloud(self) -> bool:
        return self.id.startswith('cloud')

    def make_catalog_path(self, path: str) -> str:
        """Return a catalog path for the given relative path, using this database root."""
        return f'{self.prefix}/{path}'.strip('/')


class UtilsDefinitionError(Exception):
    """A tests.utils UDF or iterator that pixeltable-rs could not define when this module was imported."""


class _Undefined:
    """Bound in place of a module-level UDF/iterator whose definition raised."""

    def __init__(self, name: str, exc: BaseException) -> None:
        self._name = name
        self._exc = exc

    def _fail(self) -> UtilsDefinitionError:
        return UtilsDefinitionError(
            f'tests.utils.{self._name} could not be defined with pixeltable-rs: {type(self._exc).__name__}: {self._exc}'
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise self._fail() from self._exc

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__'):
            raise AttributeError(name)
        raise self._fail() from self._exc


# name -> error, for module-level definitions that failed (reported by the conftest header)
DEFINITION_ERRORS: dict[str, str] = {}


def _undefined(name: str, exc: BaseException) -> Any:
    DEFINITION_ERRORS[name] = f'{type(exc).__name__}: {exc}'
    return _Undefined(name, exc)


def _testing_hooks() -> Any:
    try:
        import pixeltable._testing as hooks
    except ImportError as exc:
        raise RuntimeError(
            'pixeltable._testing is not importable; the harness needs its reset_db() and reload_catalog() hooks'
        ) from exc
    return hooks


@contextmanager
def pxt_raises(code: pxt.ErrorCode, *, match: str | None = None) -> Iterator[pytest.ExceptionInfo[pxt.Error]]:
    """Use this in place of pytest.raises() if the expected exception is a pxt.Error."""
    error_class = getattr(excs, '_error_class', None)
    cls = error_class(code) if error_class is not None else pxt.Error
    with pytest.raises(cls, match=match) as info:
        yield info
    assert info.value.error_code is code, f'expected {code.name!r}, got {info.value.error_code.name!r}'


def runs_linux_with_gpu() -> bool:
    try:
        import torch

        return sysconfig.get_platform() == 'linux-x86_64' and torch.cuda.is_available()
    except ImportError:
        return False


_PORTS = range(23000, 32768)
_port_cursor: int | None = None


def free_port() -> int:
    """Return a free loopback port for a process that binds it later (same as upstream)."""
    global _port_cursor  # noqa: PLW0603
    slice_len = len(_PORTS) // int(os.environ.get('PYTEST_XDIST_WORKER_COUNT', '1'))
    worker = int(os.environ.get('PYTEST_XDIST_WORKER', 'gw0').removeprefix('gw'))
    first = _PORTS.start + worker * slice_len
    if _port_cursor is None:
        _port_cursor = os.getpid() % slice_len
    for _ in range(slice_len):
        port = first + _port_cursor
        _port_cursor = (_port_cursor + 1) % slice_len
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
            except OSError:
                continue
        return port
    raise RuntimeError(f'No free port in {first}-{first + slice_len - 1}')


try:

    @pxt.udf
    def sleep(n: float) -> float:
        """Sleep for n seconds, return n. Used in tests to deliberately delay inserts."""
        time.sleep(n)
        return n

except Exception as _exc:
    sleep = _undefined('sleep', _exc)


try:

    @pxt.udf
    def error(error: bool) -> bool:
        """Raise AssertionError if error == True"""
        if error:
            raise AssertionError()
        return False

except Exception as _exc:
    error = _undefined('error', _exc)


# Upstream make_default_type() maps ColumnType.Type(i % 5) to a non-nullable type: STRING, INT, FLOAT, BOOL,
# TIMESTAMP. The public type hints below normalize to the same non-nullable types.
_DEFAULT_TYPES: list[Any] = [pxt.String, pxt.Int, pxt.Float, pxt.Bool, pxt.Timestamp]


def make_tbl(name: str = 'test', col_names: list[str] | None = None, is_data_versioned: bool = True) -> pxt.Table:
    if col_names is None:
        col_names = ['c1']
    schema: dict[str, Any] = {}
    for i, col_name in enumerate(col_names):
        schema[f'{col_name}'] = _DEFAULT_TYPES[i % 5]
    return pxt.create_table(name, schema, _is_data_versioned=is_data_versioned)


@dataclass(frozen=True)
class TypeInfo:
    """A column type, recovered from its public string form (`ColumnMetadata['type_']`, `ResultSet.schema`)."""

    kind: str  # 'String', 'Int', 'Float', 'Bool', 'Timestamp', 'Date', 'UUID', 'Binary', 'Json', 'Array', 'Image', ...
    nullable: bool
    base: str  # the type string without ' | None'
    shape: tuple[int | None, ...] | None = None  # Array only
    dtype: str | None = None  # Array only


_ARRAY_RE = re.compile(r'^Array\[\((?P<shape>[^)]*)\),\s*(?P<dtype>\w+)\]$')
_ARRAY_DTYPE_ONLY_RE = re.compile(r'^Array\[(?P<dtype>\w+)\]$')


def type_info(col_type: Any) -> TypeInfo:
    """Parse a column type given as its repr string (or as any object whose repr is that string)."""
    s = col_type if isinstance(col_type, str) else repr(col_type)
    s = s.strip()
    nullable = s.endswith(' | None')
    base = s.removesuffix(' | None').strip()
    m = re.match(r'[A-Za-z]+', base)
    kind = m.group(0) if m else base
    shape: tuple[int | None, ...] | None = None
    dtype: str | None = None
    if kind == 'Array':
        am = _ARRAY_RE.match(base)
        if am is not None:
            dims = [d.strip() for d in am.group('shape').split(',') if d.strip()]
            shape = tuple(None if d == 'None' else int(d) for d in dims)
            dtype = am.group('dtype')
        else:
            dm = _ARRAY_DTYPE_ONLY_RE.match(base)
            if dm is not None:
                dtype = dm.group('dtype')
    return TypeInfo(kind=kind, nullable=nullable, base=base, shape=shape, dtype=dtype)


def _installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        return False


def create_table_data(
    t: pxt.Table,
    col_names: list[str] | None = None,
    num_rows: int = 10,
    non_serializable_json: bool = False,
    arrow_compatible_json: bool = False,
) -> list[dict[str, Any]]:
    """Same as upstream; column types come from `t.get_metadata()` instead of the internal `ColumnRef.col_type`."""
    if col_names is None:
        col_names = []
    data: dict[str, Any] = {}

    serializable_json_values: list[Any] = [
        {
            'detections': [
                {
                    'id': '637e8e073b28441a453564cf',
                    'tags': ['a', 'b'],
                    'label': 'potted plant',
                    'bounding_box': [0.370, 0.335, 0.039, 0.163],
                    'mask': 'rle:0,1,0,1',
                    'confidence': 0.95,
                    'iscrowd': 0,
                    'verified': True,
                }
            ]
        },
        {'level1': {'level2': {'level3': [1, 2, 3, 4]}}},
        {'s': 'hello', 'i': 42, 'f': 2.718, 'b': False},
    ]
    if not arrow_compatible_json:
        serializable_json_values.extend(
            [
                {'level1': {'level2': {'level3': [1, 2.0, 'three', False, None]}}},
                [{'key': 'val1', 'n': 1}, {'key': 'val2', 'n': 2}],
                {'s': 'hello', 'i': 42, 'f': 2.718, 'b': False, 'n': None},
            ]
        )

    non_serializable_json_values: list[Any] = [
        {'label': 'synthetic', 'thumbnail': PIL.Image.new('RGB', (4, 4), color=(255, 0, 0))},
        {'label': 'embedding', 'vector': np.zeros(8, dtype=np.float64)},
        {'label': 'binary_payload', 'raw': b'\xde\xad\xbe\xef'},
        {
            'label': 'mixed_list',
            'items': [
                'text_item',
                42,
                3.14,
                True,
                None,
                PIL.Image.new('L', (2, 2)),
                np.ones(3, dtype=np.float32),
                b'\x00\x01',
            ],
        },
    ]

    sample_json_values = (
        serializable_json_values + non_serializable_json_values if non_serializable_json else serializable_json_values
    )

    md_columns = t.get_metadata()['columns']
    if len(col_names) == 0:
        col_names = [name for name, c in md_columns.items() if not c['is_computed']]

    col_types = {name: type_info(md_columns[name]['type_']) for name in col_names}
    for col_name in col_names:
        info = col_types[col_name]
        col_data: Any = None
        if info.kind == 'String':
            col_data = ['test string'] * num_rows
        if info.kind == 'Int':
            col_data = np.random.randint(0, 100, size=num_rows).tolist()
        if info.kind == 'Float':
            col_data = (np.random.random(size=num_rows) * 100).tolist()
        if info.kind == 'Bool':
            col_data = np.random.randint(0, 2, size=num_rows)
            col_data = [False if i == 0 else True for i in col_data]  # noqa: SIM211
        if info.kind == 'Timestamp':
            col_data = [datetime.datetime.now()] * num_rows
        if info.kind == 'Date':
            col_data = [datetime.date.today()] * num_rows
        if info.kind == 'UUID':
            col_data = [uuid.uuid4() for _ in range(num_rows)]
        if info.kind == 'Binary':
            col_data = [b'1$\x03\xfe'] * num_rows
        if info.kind == 'Json':
            col_data = [sample_json_values[i % len(sample_json_values)] for i in range(num_rows)]
        if info.kind == 'Array':
            col_data = [np.ones(info.shape, dtype=info.dtype)] * num_rows
        if info.kind == 'Image':
            image_path = get_image_files()[0]
            col_data = [image_path] * num_rows
        if info.kind == 'Video':
            video_path = get_video_files()[0]
            col_data = [video_path] * num_rows
        if info.kind == 'Audio':
            audio_path = get_audio_files()[0]
            col_data = [audio_path] * num_rows
        if info.kind == 'Document':
            docs = get_documents()
            if not _installed('mistune') or not _installed('markitdown'):
                skip_exts = {'.md', '.pptx', '.docx', '.xlsx'}
                docs = [d for d in docs if Path(d).suffix.lower() not in skip_exts]
            col_data = [docs[i % len(docs)] for i in range(num_rows)]

        data[col_name] = col_data
    rows = [{col_name: data[col_name][i] for col_name in col_names} for i in range(num_rows)]
    return rows


def create_test_tbl(name: str = 'test_tbl', is_data_versioned: bool = True) -> pxt.Table:
    schema: dict[str, Any] = {
        'c1': {'type': pxt.String, 'comment': 'String column with no nulls'},
        'c1n': {'type': pxt.String | None, 'custom_metadata': {'nullable': True}},
        'c2': pxt.Int,
        'c3': pxt.Float,
        'c4': pxt.Bool,
        'c5': pxt.Timestamp,
        'c6': pxt.Json,
        'c7': pxt.Json,
    }
    t = pxt.create_table(name, schema, primary_key='c2', _is_data_versioned=is_data_versioned)
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


def create_img_tbl(name: str = 'test_img_tbl', num_rows: int = 0, is_data_versioned: bool = True) -> pxt.Table:
    schema = {'img': pxt.Image, 'category': pxt.String, 'split': pxt.String}
    tbl = pxt.create_table(name, schema, _is_data_versioned=is_data_versioned)
    rows = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
    if num_rows > 0:
        # select output_rows randomly in the hope of getting a good sample of the available categories
        rng = np.random.default_rng(17)
        idxs = rng.choice(np.arange(len(rows)), size=num_rows, replace=False)
        rows = [rows[i] for i in idxs]
    tbl.insert(rows)
    return tbl


def assert_schema_eq(actual: pxt.Table, expected: pxt.Table) -> None:
    """Assert two tables have the same column schema (names and types), via the public get_metadata()."""

    def col_types(tbl: pxt.Table) -> dict[str, str]:
        return {name: col['type_'] for name, col in tbl.get_metadata()['columns'].items()}

    a, e = col_types(actual), col_types(expected)
    assert a == e, f'schema mismatch:\n  actual:   {a}\n  expected: {e}'


try:
    ALL_DATATYPES_SCHEMA: dict[str, Any] = {
        'row_id': pxt.Int,
        'c_array': pxt.Array[(10,), pxt.Float] | None,
        'c_audio': pxt.Audio | None,
        'c_bool': pxt.Bool | None,
        'c_date': pxt.Date | None,
        'c_float': pxt.Float | None,
        'c_image': pxt.Image | None,
        'c_int': pxt.Int | None,
        'c_json': pxt.Json | None,
        'c_string': pxt.String | None,
        'c_timestamp': pxt.Timestamp | None,
        'c_uuid': pxt.UUID | None,
        'c_binary': pxt.Binary | None,
        'c_video': pxt.Video | None,
        'c_document': pxt.Document | None,
    }
except Exception as _exc:
    ALL_DATATYPES_SCHEMA = _undefined('ALL_DATATYPES_SCHEMA', _exc)


def create_all_datatypes_tbl(
    name: str = 'all_datatype_tbl',
    non_serializable_json: bool = False,
    arrow_compatible_json: bool = False,
    is_data_versioned: bool = True,
) -> pxt.Table:
    """Creates a table with all supported datatypes."""
    tbl = pxt.create_table(name, ALL_DATATYPES_SCHEMA, _is_data_versioned=is_data_versioned)
    example_rows = create_table_data(
        tbl, num_rows=11, non_serializable_json=non_serializable_json, arrow_compatible_json=arrow_compatible_json
    )

    for i, r in enumerate(example_rows):
        r['row_id'] = i  # row_id

    tbl.insert(example_rows)
    return tbl


def create_scalars_tbl(num_rows: int, seed: int = 0, percent_nulls: int = 10, path: str = 'scalars_tbl') -> pxt.Table:
    """
    Creates a table with scalar columns, each of which contains randomly generated data.
    (Upstream builds the schema from ts.*Type(nullable=...); the public type hints below are the same types.)
    """
    assert percent_nulls >= 0 and percent_nulls <= 100
    rng = np.random.default_rng(seed)
    schema = {
        'row_id': pxt.Int,  # used for row selection
        'c_bool': pxt.Bool | None,
        'c_float': pxt.Float | None,
        'c_int': pxt.Int | None,
        'c_string': pxt.String | None,
        'c_timestamp': pxt.Timestamp | None,
    }
    tbl = pxt.create_table(path, schema)

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


def get_video_files(
    include_bad_video: bool = False, include_vfr: bool = True, include_mpgs: bool = True, extension: str | None = None
) -> list[str]:
    glob_result = glob.glob(f'{TESTS_DIR}/**/videos/*', recursive=True)
    if not include_bad_video:
        glob_result = [f for f in glob_result if 'bad_video' not in f]
    if not include_vfr:
        glob_result = [f for f in glob_result if 'vfr' not in f]
    if not include_mpgs:
        glob_result = [f for f in glob_result if not f.endswith('.mpg')]
    if extension is not None:
        glob_result = [f for f in glob_result if f.endswith(extension)]
    glob_result.sort()
    return glob_result


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
    """Create test video with specific properties using ffmpeg and return its path.
    (Upstream also passes Env.default_video_encoder; this uses ffmpeg's default encoder.)"""
    cmd = ['ffmpeg', '-f', 'lavfi', '-i', f'testsrc=duration={duration}:size={size}:rate={fps}']
    if has_audio:
        cmd.extend(['-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}'])
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
        modes = sorted({mode for mode, _ in good_images})
        groups = [[f for mode, f in good_images if mode == mode_group] for mode_group in modes]
        for group in groups:
            group.sort()
            random.Random(4171780).shuffle(group)
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
    """Upstream lists s3://multimedia-commons/data/videos/mp4/ through Pixeltable's internal ObjectOps; this lists it
    with an anonymous boto3 client, and skips the test when boto3 is missing or S3 is unreachable."""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config as BotoConfig
    except ImportError:
        pytest.skip('boto3 is not installed (needed to list s3://multimedia-commons)')
    try:
        client = boto3.client('s3', config=BotoConfig(signature_version=UNSIGNED))
        resp = client.list_objects_v2(Bucket='multimedia-commons', Prefix='data/videos/mp4/', MaxKeys=n)
    except Exception as exc:
        pytest.skip(f's3://multimedia-commons is not reachable: {exc}')
    return [f's3://multimedia-commons/{obj["Key"]}' for obj in resp.get('Contents', [])][:n]


def get_audio_files(include_bad_audio: bool = False, extension: str | None = None) -> list[str]:
    audio_dir = TESTS_DIR / 'data' / 'audio'
    glob_result = glob.glob(f'{audio_dir}/*', recursive=True)
    if not include_bad_audio:
        glob_result = [f for f in glob_result if 'bad_audio' not in f]
    if extension is not None:
        glob_result = [f for f in glob_result if f.endswith(extension)]
    glob_result.sort()
    return glob_result


def get_audio_file(name: str) -> str | None:
    audio_dir = TESTS_DIR / 'data' / 'audio'
    file_path = audio_dir / name
    glob_result = glob.glob(f'{file_path}', recursive=True)
    glob_result.sort()
    return glob_result.pop(0) if len(glob_result) > 0 else None


def get_documents() -> list[str]:
    docs_dir = TESTS_DIR / 'data' / 'documents'
    glob_result = glob.glob(f'{docs_dir}/*', recursive=True)
    glob_result.sort()
    return glob_result


def get_csv_files() -> list[str]:
    csv_dir = TESTS_DIR / 'data' / 'csv'
    glob_result = glob.glob(f'{csv_dir}/*', recursive=True)
    glob_result.sort()
    return glob_result


def get_csv_file(name: str) -> str | None:
    csv_dir = TESTS_DIR / 'data' / 'csv'
    file_path = csv_dir / name
    glob_result = glob.glob(f'{file_path}', recursive=True)
    glob_result.sort()
    return glob_result.pop(0) if len(glob_result) > 0 else None


def get_json_files() -> list[str]:
    json_dir = TESTS_DIR / 'data' / 'json'
    glob_result = glob.glob(f'{json_dir}/*', recursive=True)
    glob_result.sort()
    return glob_result


def get_json_file(name: str) -> str | None:
    json_dir = TESTS_DIR / 'data' / 'json'
    file_path = json_dir / name
    glob_result = glob.glob(f'{file_path}', recursive=True)
    return glob_result.pop(0) if len(glob_result) > 0 else None


def get_sentences(n: int = 100) -> list[str]:
    path = glob.glob(f'{TESTS_DIR}/**/jeopardy.json', recursive=True)[0]
    with open(path, 'r', encoding='utf8') as f:
        questions_list = json.load(f)
    return [q['question'].replace("'", '') for q in questions_list[:n]]


def assert_type_eq(col_type: Any, pxt_type: Any) -> None:
    """Upstream: `col_type == ts.ColumnType.normalize_type(pxt_type)`. pixeltable.type_system is public in
    pixeltable-rs, so this keeps the upstream comparison and fails clearly if the module lacks it."""
    from pixeltable import type_system as ts

    assert col_type == ts.ColumnType.normalize_type(pxt_type)


def _schema_types(rs: pxt.ResultSet) -> dict[str, TypeInfo]:
    """Column types of a result set, from the public `ResultSet.schema` (column name -> type string).
    Upstream reads the internal `ResultSet._schema` (column name -> ColumnType)."""
    return {name: type_info(t) for name, t in rs.schema.items()}


def assert_resultset_eq(
    r1: pxt.ResultSet, r2: pxt.ResultSet, compare_col_names: bool = False, compare_col_types: bool = True
) -> None:
    assert len(r1) == len(r2)
    assert len(r1.schema) == len(r2.schema)
    types1, types2 = _schema_types(r1), _schema_types(r2)
    if compare_col_types:
        # ColumnType.matches(): equal aside from nullability
        assert all(t1.base == t2.base for t1, t2 in zip(types1.values(), types2.values()))
    if compare_col_names:
        assert r1.schema.keys() == r2.schema.keys()
    for r1_col, r2_col in zip(r1.schema, r2.schema):
        assert_columns_eq(r1_col, types1[r1_col], r1[r1_col], r2[r2_col])


def assert_columns_eq(col_name: str, col_type: Any, c1: list[Any], c2: list[Any]) -> None:
    """col_type may be a ColumnType, its string form, or a TypeInfo; only its kind selects the comparer."""
    info = col_type if isinstance(col_type, TypeInfo) else type_info(col_type)
    mismatches = __find_column_mismatches(info.kind, c1, c2)
    assert len(mismatches) == 0, __mismatch_err_string(col_name, c1, c2, mismatches)


def __find_column_mismatches(kind: str, s1: list[Any], s2: list[Any]) -> list[int]:
    comparer = __COMPARERS.get(kind, __equality_comparer)
    mismatches = []
    for i, (v1, v2) in enumerate(zip(s1, s2)):
        if (v1 is None) != (v2 is None) or (v1 is not None and not comparer(v1, v2)):
            mismatches.append(i)
    return mismatches


def __float_comparer(x: float, y: float) -> bool:
    return bool(np.isclose(x, y, equal_nan=True))


def __array_comparer(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(x, y)


def sha256sum(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def __file_comparer(x: str, y: str) -> bool:
    return sha256sum(x) == sha256sum(y)


def __equality_comparer(x: Any, y: Any) -> bool:
    return x == y


__IMAGE_PHASH_TOLERANCE = 4


def __image_comparer(x: PIL.Image.Image, y: PIL.Image.Image) -> bool:
    """Perceptual-hash comparison: tolerates lossy re-encoding while still rejecting visually distinct images."""
    import imagehash

    if x.size != y.size or x.mode != y.mode:
        return False
    return (imagehash.phash(x) - imagehash.phash(y)) <= __IMAGE_PHASH_TOLERANCE


def __json_comparer(x: Any, y: Any) -> bool:
    if isinstance(x, PIL.Image.Image) and isinstance(y, PIL.Image.Image):
        return __image_comparer(x, y)
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


__COMPARERS: dict[str, Callable[[Any, Any], bool]] = {
    'Float': __float_comparer,
    'Array': __array_comparer,
    'Json': __json_comparer,
    'Image': __image_comparer,
    'Video': __file_comparer,
    'Audio': __file_comparer,
    'Document': __file_comparer,
}


def assert_dicts_eq(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Assert two dicts are equal, comparing arrays elementwise and floats with `np.isclose`."""
    assert __json_comparer(actual, expected), f'{actual!r} != {expected!r}'


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
    `version_created` will be checked to be less than 1 minute ago; `id` is asserted to be a UUID but its value
    is not compared; the other fields will be checked for exact equality.
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    actual_created_at: datetime.datetime = actual['version_created']
    assert (now - actual_created_at).total_seconds() <= 60
    assert isinstance(actual['id'], uuid.UUID)

    trimmed_actual = {k: v for k, v in actual.items() if k not in {'version_created', 'id'}}
    tc = TestCase()
    tc.maxDiff = 25_000
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
    tc.maxDiff = 25_000
    tc.assertDictEqual(expected, trimmed_actual)


def strip_lines(s: str) -> str:
    lines = s.split('\n')
    return '\n'.join(line.strip() for line in lines)


def skip_test_if_not_installed(*packages: str) -> None:
    """Upstream asks Env.is_installed_package(); this asks importlib, which is what that does."""
    for package in packages:
        if not _installed(package):
            pytest.skip(f'Package `{package}` is not installed.')


def skip_test_if_not_in_path(*binaries: str) -> None:
    for binary in binaries:
        if shutil.which(binary) is None:
            pytest.skip(f'Binary `{binary}` is not in PATH.')


def skip_test_if_not_local(reason: str = 'Requires a local (file-backed) Pixeltable database.') -> None:
    """The harness database is always local."""


def skip_test_if_no_aws_credentials() -> None:
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except ImportError:
        pytest.skip('boto3 is not installed')

    try:
        cl = boto3.client('s3')
        cl.list_buckets()
    except NoCredentialsError as exc:
        pytest.skip(str(exc).splitlines()[0])


def validate_update_status(status: pxt.UpdateStatus, expected_rows: int | None = None) -> None:
    assert status.num_excs == 0
    if expected_rows is not None:
        assert status.num_rows == expected_rows, status


def run_workers(target: Callable[[int], None], n_threads: int) -> list[tuple[int, BaseException]]:
    """Run target(tid) on n_threads workers; return any exceptions raised."""
    errors: list[tuple[int, BaseException]] = []
    lock = threading.Lock()

    def runner(tid: int) -> None:
        try:
            target(tid)
        except BaseException as e:
            with lock:
                errors.append((tid, e))

    threads = [threading.Thread(target=runner, args=(i,)) for i in range(n_threads)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    return errors


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
    read_table = parquet.read_table(table_path)
    assert read_table.num_rows == test_table.num_rows
    assert set(read_table.column_names) == set(test_table.column_names)
    assert read_table.schema.equals(test_table.schema)
    assert read_table.equals(test_table)
    return table_path


def reload_catalog(reload: bool = True) -> None:
    """Drop all in-memory catalog state; the next call reloads it from disk.
    Upstream: reset_runtime(); pxt.init(). Here: the pixeltable._testing.reload_catalog() hook."""
    if not reload:
        return
    _testing_hooks().reload_catalog()


@contextmanager
def capture_console_output(match: str | None = None) -> Iterator[StringIO]:
    """Capture what Pixeltable logs to the 'pixeltable' logger. (Upstream filters through its internal
    ConsoleOutputHandler/ConsoleMessageFilter; this captures every record at DEBUG and above.)"""
    pxt_logger = logging.getLogger('pixeltable')
    sio = StringIO()
    handler = logging.StreamHandler(stream=sio)
    handler.setLevel(10)
    try:
        pxt_logger.addHandler(handler)
        yield sio
        if match is not None:
            contents = sio.getvalue()
            assert re.search(match, contents) is not None, (
                f'Console output did not match.\nRegex: {match!r}\nActual: {contents}'
            )
    finally:
        pxt_logger.removeHandler(handler)
        sio.flush()


try:

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
            # Return 0.0 instead of None, to distinguish between these two cases: the tool not being called, and the
            # tool being called on a symbol other than NVDA/UAL
            return 0.0

except Exception as _exc:
    stock_price = _undefined('stock_price', _exc)


# Local path to a sample image to use in tests. Use it whenever possible instead of fetching files over the network.
SAMPLE_IMAGE_FILE_PATH = str(TESTS_DIR.parent / 'docs' / 'resources' / 'images' / '000000000009.jpg')
SAMPLE_IMAGE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000009.jpg'


class ReloadTester:
    """Utility to verify that queries return identical results after a catalog reload"""

    query_info: list[tuple[dict[str, Any], pxt.ResultSet]]  # list of (query.as_dict(), query.collect())

    def __init__(self) -> None:
        self.query_info = []

    def clear(self) -> None:
        self.query_info = []

    def run_query(self, query: pxt.Query) -> pxt.ResultSet:
        query_dict = query.as_dict()
        result_set = query.collect()
        self.query_info.append((query_dict, result_set))
        return result_set

    def run_reload_test(self, clear: bool = True) -> None:
        reload_catalog()
        assert len(self.query_info) > 0, 'No queries in ReloadTester!'
        # enumerate(): the list index is useful for debugging
        for idx, (query_dict, result_set) in enumerate(self.query_info):
            # upstream wraps from_dict() in catalog.retry_loop(), which retries Postgres serialization failures
            query = pxt.Query.from_dict(query_dict)

            new_result_set = query.collect()
            try:
                assert_resultset_eq(result_set, new_result_set, compare_col_names=True)
            except Exception as e:
                s = f'Reload test failed for query:\nIndex: {idx}\n{query}\n{e}'
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
        kwargs['only_rerun'] = []
    return pytest.mark.flaky(**kwargs)


NETWORK_ERROR_PATTERNS = [
    '429',
    '503',
    'Too Many Requests',
    'Connection reset',
    'Connection aborted',
    'Service unavailable',
    'timed out',
    'Timeout',
    'ExternalServiceError',
    'URLError',
    'Remote end closed connection',
]


def rerun_on_network_error(reruns: int = 3, reruns_delay: int = 15, **kwargs: Any) -> Callable:
    """Rerun a test that fails due to a transient network/service error (needs pytest-rerunfailures to take effect)."""
    return rerun(reruns=reruns, reruns_delay=reruns_delay, only_rerun=NETWORK_ERROR_PATTERNS, **kwargs)


IN_CI = bool(os.environ.get('PXTTEST_IN_CI'))
CI_OS = os.environ.get('PXTTEST_CI_OS')


class DummyIteratorOut(TypedDict):
    out1: str
    out2: int


try:

    @pxt.iterator
    class DummyIterator(pxt.PxtIterator[DummyIteratorOut]):
        count: int
        limit: int

        def __init__(self, limit: int) -> None:
            self.count = 0
            self.limit = limit

        def __next__(self) -> DummyIteratorOut:
            if self.count >= self.limit:
                raise StopIteration
            result = DummyIteratorOut(out1=f'str{self.count}', out2=self.count)
            self.count += 1
            return result

except Exception as _exc:
    DummyIterator = _undefined('DummyIterator', _exc)


class DummyIterator2Out(TypedDict):
    out1: str
    out3: int


try:

    @pxt.iterator
    class DummyIterator2(pxt.PxtIterator[DummyIterator2Out]):
        count: int
        limit: int

        def __init__(self, limit: int) -> None:
            self.count = 0
            self.limit = limit

        def __next__(self) -> DummyIterator2Out:
            if self.count >= self.limit:
                raise StopIteration
            result = DummyIterator2Out(out1=f'str{self.count}', out3=self.count)
            self.count += 1
            return result

except Exception as _exc:
    DummyIterator2 = _undefined('DummyIterator2', _exc)


def _process_lifetime_loop_ids() -> set[int]:
    module = sys.modules.get('lancedb.background_loop')
    if module is None:
        return set()
    loop = getattr(module.LOOP, 'loop', None)
    return set() if loop is None else {id(loop)}


def open_async_resources() -> list[str]:
    """Describes every event loop and HTTP client session that is still open, one string each."""
    gc.collect()
    ignored_loop_ids = _process_lifetime_loop_ids()
    resources: list[str] = []
    aiohttp = sys.modules.get('aiohttp')
    httpx = sys.modules.get('httpx')
    for obj in gc.get_objects():
        if isinstance(obj, asyncio.AbstractEventLoop) and not obj.is_closed() and id(obj) not in ignored_loop_ids:
            resources.append(f'event loop {type(obj).__name__} at {id(obj):#x}')
        elif aiohttp is not None and isinstance(obj, aiohttp.ClientSession) and not obj.closed:
            resources.append(f'aiohttp session at {id(obj):#x}')
        elif httpx is not None and isinstance(obj, httpx.AsyncClient) and not obj.is_closed:
            resources.append(f'httpx client at {id(obj):#x}')
    return resources


def validate_async_teardown() -> list[str]:
    """Upstream retires the runtime (reset_runtime()) and asserts that no event loop or HTTP client session is left
    open. Here the catalog is dropped through the reload hook and the leaks are returned; the conftest reports them
    as a warning rather than failing whichever test happened to run last."""
    try:
        _testing_hooks().reload_catalog()
    except Exception:
        pass
    return open_async_resources()


# id(handle) -> (handle, table id): a dropped table's handle can no longer answer get_metadata()
_TABLE_IDS: dict[int, tuple[pxt.Table, uuid.UUID]] = {}


def _table_id(tbl: pxt.Table) -> uuid.UUID:
    try:
        tbl_id = uuid.UUID(str(tbl.get_metadata()['id']))
    except pxt.Error:
        if id(tbl) in _TABLE_IDS:
            return _TABLE_IDS[id(tbl)][1]
        raise
    _TABLE_IDS[id(tbl)] = (tbl, tbl_id)
    return tbl_id


def check_media_store_count(
    tbl: pxt.Table,
    expected_count: int,
    db_root: DatabaseRoot,
    *,
    tbl_version: int | None = None,
    default_input_dest: bool = False,
    default_output_dest: bool = False,
) -> None:
    """Assert the number of media files stored for tbl.

    Upstream counts through its internal ObjectOps, keyed by the internal `tbl._id`. This counts files in
    `$PIXELTABLE_HOME/media` (recursively) whose names follow upstream's media file naming,
    `<table id hex>_<column id>_<table version>_<unique id>.<ext>`, with the table id from the public
    `get_metadata()['id']` (remembered per handle, since the tests also count after dropping the table). It therefore
    holds pixeltable-rs to upstream's media layout under the home directory. default_input_dest/default_output_dest
    select a destination upstream; the harness never configures one, so both mean the home media dir.
    """
    if db_root.is_cloud:
        return
    prefix = _table_id(tbl).hex
    media_dir = Path(os.environ['PIXELTABLE_HOME']) / 'media'
    pattern = f'{prefix}_*' if tbl_version is None else f'{prefix}_*_{tbl_version}_*'
    actual = sum(1 for p in media_dir.rglob(pattern) if p.is_file()) if media_dir.is_dir() else 0
    assert actual == expected_count, f'expected {expected_count} media objects, found {actual}'


def assert_image_bytes(data: bytes, *, size: tuple[int, int] | None = None) -> None:
    """Decode data as an image; optionally assert dimensions."""
    PIL.Image.open(io.BytesIO(data)).verify()
    if size is not None:
        assert PIL.Image.open(io.BytesIO(data)).size == size


def validate_repr(t: Any, expected: str) -> None:
    def cleanup(r: str) -> str:
        r = re.sub(r'-{3,}', '---', r)  # normalize separator lines
        r = re.sub(r'\.{3,}', '...', r)
        return re.sub(r'\s+', ' ', r).strip()  # normalize whitespace

    assert cleanup(repr(t)) == cleanup(expected), f'Expected repr: {expected}, actual: {t!r}'
    t._repr_html_()  # the IPython display protocol, part of the public surface


try:

    @pxt.udf
    def dummy_embedding(text: str, n: int) -> pxt.Array[(None,), np.float32]:
        if 'zero' in text:
            arr = np.zeros((n,), dtype=np.float32)
            arr[n // 2 :] = 1
            return arr
        if 'one' in text:
            arr = np.zeros((n,), dtype=np.float32)
            arr[: n // 2] = 1
            return arr
        return np.random.rand(n).astype(np.float32)

    @dummy_embedding.overload
    def _(img: PIL.Image.Image, n: int) -> pxt.Array[(None,), np.float32]:
        return np.random.rand(n).astype(np.float32)

    @dummy_embedding.conditional_return_type
    def _(n: int) -> Any:
        from pixeltable import type_system as ts

        return ts.ArrayType((n,), dtype=np.dtype('float32'), nullable=False)

except Exception as _exc:
    dummy_embedding = _undefined('dummy_embedding', _exc)


# A deterministic, download-free embedding used in place of HuggingFace models (clip, sentence_transformer) in tests
# that exercise embedding-index machinery rather than real-model semantics (same as upstream).

LOCAL_EMBED_DIM = 512  # matches openai/clip-vit-base-patch32; bind via local_embedding.using(dim=...)


def _local_text_vec(text: str, dim: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    t = (text or '').lower()
    tokens = list(t) + [t[i : i + 3] for i in range(len(t) - 2)]  # unigrams + character trigrams
    for tok in tokens:
        v[int.from_bytes(hashlib.sha1(tok.encode('utf-8')).digest()[:4], 'big') % dim] += 1.0
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _local_image_vec(image: PIL.Image.Image, dim: int) -> np.ndarray:
    side = math.isqrt(dim)
    grid = image.convert('L').resize((-(-dim // side), side))  # downsample to a grid of >= dim pixels
    v = np.asarray(grid, dtype=np.float32).reshape(-1)[:dim]
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


try:

    @pxt.udf
    def local_embedding(text: str, *, dim: int = LOCAL_EMBED_DIM) -> pxt.Array[(None,), np.float32]:
        return _local_text_vec(text, dim)

    @local_embedding.overload
    def _(image: PIL.Image.Image, *, dim: int = LOCAL_EMBED_DIM) -> pxt.Array[(None,), np.float32]:
        return _local_image_vec(image, dim)

    @local_embedding.conditional_return_type
    def _(dim: int) -> Any:
        from pixeltable import type_system as ts

        return ts.ArrayType((dim,), dtype=np.dtype('float32'), nullable=False)

except Exception as _exc:
    local_embedding = _undefined('local_embedding', _exc)


def btree_idxs(t: pxt.Table) -> dict[str, str]:
    """The names of `t`'s B-tree indexes, mapped to the column each one indexes."""
    btree_md = [info for info in t.get_metadata()['indexes'].values() if info['index_type'] == 'btree']
    assert all(len(info['columns']) == 1 for info in btree_md)
    return {info['name']: info['columns'][0] for info in btree_md}


def schema_from_tbl_md(metadata: pxt.TableMetadata) -> dict[str, Any]:
    return {
        'kind': metadata['kind'],
        'comment': metadata['comment'],
        'custom_metadata': metadata['custom_metadata'],
        'media_validation': metadata['media_validation'],
        'primary_key': metadata['primary_key'],
        'iterator_call': metadata['iterator_call'],
        'columns': {
            name: {
                'type_': info['type_'],
                'is_stored': info['is_stored'],
                'is_primary_key': info['is_primary_key'],
                'media_validation': info['media_validation'],
                'is_computed': info['is_computed'],
                'computed_with': info['computed_with'],
                'is_builtin': info['is_builtin'],
                'comment': info['comment'],
                'custom_metadata': info['custom_metadata'],
                'is_iterator_col': info['is_iterator_col'],
                'destination': info['destination'],
            }
            for name, info in metadata['columns'].items()
        },
        'indexes': metadata['indexes'],
    }


def _internal_only(name: str, what: str) -> Callable[..., Any]:
    def helper(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f'tests.utils.{name}() is not available in the pixeltable-rs harness: upstream implements it with {what}'
        )

    helper.__name__ = name
    return helper


# Upstream helpers that only exist in terms of Pixeltable internals or infrastructure the harness does not have.
# None of the selected test files import them; they fail clearly if a newly selected file does.
list_store_indexes = _internal_only('list_store_indexes', "the table's Postgres store (pg_indexes)")
get_temp_store_count = _internal_only('get_temp_store_count', 'the internal TempStore')
check_temp_store_count = _internal_only('check_temp_store_count', 'the internal TempStore')
reload_env = _internal_only('reload_env', 'Config/Env re-initialization')
skip_test_if_no_client = _internal_only('skip_test_if_no_client', 'the runtime client registry')
skip_test_if_no_config = _internal_only('skip_test_if_no_config', 'the internal Config')
skip_test_if_no_pxt_credentials = _internal_only('skip_test_if_no_pxt_credentials', 'the management client')
home_bucket_uri = _internal_only('home_bucket_uri', 'catalog Path parsing for hosted databases')
iceberg_catalog = _internal_only('iceberg_catalog', 'Env.require_package()')
aux_db_name = _internal_only('aux_db_name', 'the Postgres test database name')

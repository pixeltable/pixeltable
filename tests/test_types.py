# mypy: disable-error-code="misc"
# ruff: noqa: RUF031

import datetime
import json
import typing
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import PIL.Image
import pydantic
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs, exprs
from pixeltable.type_system import (
    UUID,
    Array,
    ArrayType,
    Audio,
    AudioType,
    Binary,
    BinaryType,
    Bool,
    BoolType,
    ColumnType,
    Date,
    DateType,
    Document,
    DocumentType,
    Float,
    FloatType,
    Image,
    ImageType,
    Int,
    IntType,
    Json,
    JsonType,
    Required,
    String,
    StringType,
    Timestamp,
    TimestampType,
    UUIDType,
    Video,
    VideoType,
)

from .utils import pxt_raises

FLOAT32 = np.dtype('float32')


class TypedDict1(TypedDict):
    a: str
    b: int | None
    c: Array[(None,), np.float32]


class TypedDict2(TypedDict, total=False):
    a: str
    b: int | None
    c: Array[(None,), np.float32]


class TypedDict3(TypedDict):
    a: str
    b: TypedDict1  # nested TypedDict
    c: tuple[int, ...]
    d: Tuple[int, str]  # Python 3.8-style Tuple
    e: list[int]
    f: List[int]  # Python 3.8-style List


class TypedDict4(TypedDict, total=False):
    a: typing.Required[int]  # explicit Required field marker (value type is Int; key is required)
    b: typing.NotRequired[str]  # explicit NotRequired field marker
    c: float  # optional by total=False


class Model1(pydantic.BaseModel):
    a: str
    b: list[int] | None
    c: tuple[str, ...]
    d: int | None = None


class TestTypes:
    def test_infer(self, init_env: None) -> None:
        test_cases: list[tuple[Any, ColumnType]] = [
            ('a', StringType()),
            (1, IntType()),
            (1.0, FloatType()),
            (True, BoolType()),
            (datetime.date.today(), DateType()),
            (datetime.datetime.now(), TimestampType()),
            (uuid.uuid4(), UUIDType()),
            (b'ab\x03\xfe', BinaryType()),
            (PIL.Image.new('RGB', (100, 100)), ImageType(height=100, width=100, mode='RGB')),
            (np.ndarray((1, 2, 3), dtype=np.int64), ArrayType((1, 2, 3), dtype=np.dtype('int64'))),
            ({'a': 1, 'b': '2'}, JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}, closed=True))),
            (['3', 4], JsonType(JsonType.TypeSchema([StringType(), IntType()]))),
        ]
        for val, expected_type in test_cases:
            assert ColumnType.infer_literal_type(val) == expected_type, val

    def test_from_python_type(self, init_env: None) -> None:
        from .module_with_future_annotations import FutureQualifiedTypedDict, FutureTypedDict

        # Test cases: (python_type, pxt_type, str(pxt_type))
        test_cases: tuple[tuple[Any, ColumnType, str], ...] = (
            # Builtin and standard types
            (str, StringType(nullable=False), 'String'),
            (int, IntType(nullable=False), 'Int'),
            (float, FloatType(nullable=False), 'Float'),
            (bool, BoolType(nullable=False), 'Bool'),
            (datetime.datetime, TimestampType(nullable=False), 'Timestamp'),
            (datetime.date, DateType(nullable=False), 'Date'),
            (uuid.UUID, UUIDType(nullable=False), 'UUID'),
            (bytes, BinaryType(nullable=False), 'Binary'),
            (list, JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False), 'Json[(Json, ...)]'),
            (tuple, JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False), 'Json[(Json, ...)]'),
            (dict, JsonType(nullable=False), 'Json'),
            (list[int], JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=False), 'Json[(Int, ...)]'),
            (
                list[dict[str, int]],
                JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False),
                'Json[(Json, ...)]',
            ),
            (dict[int, str], JsonType(nullable=False), 'Json'),
            (dict[dict[str, int], list[int]], JsonType(nullable=False), 'Json'),
            (List, JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False), 'Json[(Json, ...)]'),
            (Tuple, JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False), 'Json[(Json, ...)]'),
            (Dict, JsonType(nullable=False), 'Json'),
            (List[int], JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=False), 'Json[(Int, ...)]'),
            (
                List[Dict[str, int]],
                JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False),
                'Json[(Json, ...)]',
            ),
            (Dict[int, str], JsonType(nullable=False), 'Json'),
            (PIL.Image.Image, ImageType(nullable=False), 'Image'),
            # Pixeltable types
            (String, StringType(nullable=False), 'String'),
            (Int, IntType(nullable=False), 'Int'),
            (Float, FloatType(nullable=False), 'Float'),
            (Bool, BoolType(nullable=False), 'Bool'),
            (Timestamp, TimestampType(nullable=False), 'Timestamp'),
            (Date, DateType(nullable=False), 'Date'),
            (UUID, UUIDType(nullable=False), 'UUID'),
            (Binary, BinaryType(nullable=False), 'Binary'),
            (Array, ArrayType(nullable=False), 'Array'),
            (Json, JsonType(nullable=False), 'Json'),
            (Image, ImageType(height=None, width=None, mode=None, nullable=False), 'Image'),
            (Video, VideoType(nullable=False), 'Video'),
            (Audio, AudioType(nullable=False), 'Audio'),
            (Document, DocumentType(nullable=False), 'Document'),
            # Pixeltable types with specialized parameters
            (Array[Int], ArrayType(dtype=IntType(), nullable=False), 'Array[int64]'),
            (Array[(None,), Int], ArrayType((None,), dtype=IntType(), nullable=False), 'Array[(None,), int64]'),
            (Array[(5,), Bool], ArrayType((5,), dtype=BoolType(), nullable=False), 'Array[(5,), bool]'),
            (
                Array[(5, None, 3), Float],
                ArrayType((5, None, 3), dtype=FloatType(), nullable=False),
                'Array[(5, None, 3), float32]',
            ),
            # Json list or tuple
            (Json[list[int]], JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            (Json[tuple[int, str]], JsonType(JsonType.TypeSchema([IntType(), StringType()])), 'Json[(Int, String)]'),
            (Json[tuple[int, ...]], JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            (Json[List[int]], JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            (Json[Tuple[int, str]], JsonType(JsonType.TypeSchema([IntType(), StringType()])), 'Json[(Int, String)]'),
            # Json TypedDict
            (
                Json[TypedDict1],
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(nullable=True), 'c': ArrayType((None,), dtype=FloatType())}
                    )
                ),
                "Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}]",
            ),
            (
                Json[TypedDict2],
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(nullable=True), 'c': ArrayType((None,), dtype=FloatType())},
                        optional_keys=frozenset(('a', 'b', 'c')),
                    )
                ),
                "Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}, optional_keys=['a', 'b', 'c']]",
            ),
            (
                Json[TypedDict3],
                JsonType(
                    JsonType.TypeSchema(
                        {
                            'a': StringType(),
                            'b': JsonType(
                                JsonType.TypeSchema(
                                    {
                                        'a': StringType(),
                                        'b': IntType(nullable=True),
                                        'c': ArrayType((None,), dtype=FloatType()),
                                    }
                                )
                            ),
                            'c': JsonType(JsonType.TypeSchema([], variadic_type=IntType())),
                            'd': JsonType(JsonType.TypeSchema([IntType(), StringType()])),
                            'e': JsonType(JsonType.TypeSchema([], variadic_type=IntType())),
                            'f': JsonType(JsonType.TypeSchema([], variadic_type=IntType())),
                        }
                    )
                ),
                "Json[{'a': String, 'b': Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}], "
                "'c': Json[(Int, ...)], 'd': Json[(Int, String)], 'e': Json[(Int, ...)], 'f': Json[(Int, ...)]}]",
            ),
            # Json Pydantic Models
            (
                Json[Model1],
                JsonType(
                    JsonType.TypeSchema(
                        {
                            'a': StringType(),
                            'b': JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=True),
                            'c': JsonType(JsonType.TypeSchema([], variadic_type=StringType())),
                            'd': IntType(nullable=True),
                        },
                        optional_keys=frozenset(('d',)),
                    )
                ),
                "Json[{'a': String, 'b': Json[(Int, ...)] | None, 'c': Json[(String, ...)], 'd': Int | None}, "
                "optional_keys=['d']]",
            ),
            # Required/NotRequired field markers resolve to the inner type; only the required field stays out of
            # optional_keys
            (
                Json[TypedDict4],
                JsonType(
                    JsonType.TypeSchema(
                        {'a': IntType(), 'b': StringType(), 'c': FloatType()}, optional_keys=frozenset(('b', 'c'))
                    )
                ),
                "Json[{'a': Int, 'b': String, 'c': Float}, optional_keys=['b', 'c']]",
            ),
            # TypedDicts declared under `from __future__ import annotations`
            (
                Json[FutureTypedDict],
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(nullable=True), 'img': ImageType(), 'note': StringType()},
                        optional_keys=frozenset(('note',)),
                    )
                ),
                "Json[{'a': String, 'b': Int | None, 'img': Image, 'note': String}, optional_keys=['note']]",
            ),
            (
                Json[FutureQualifiedTypedDict],
                JsonType(
                    JsonType.TypeSchema(
                        {'a': IntType(), 'b': StringType(), 'c': FloatType()}, optional_keys=frozenset(('b', 'c'))
                    )
                ),
                "Json[{'a': Int, 'b': String, 'c': Float}, optional_keys=['b', 'c']]",
            ),
            # Json "convenience structures"
            (Json[[int]], JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            (Json[(int,)], JsonType(JsonType.TypeSchema([IntType()])), 'Json[(Int,)]'),
            (
                Json[(int, str, float)],
                JsonType(JsonType.TypeSchema([IntType(), StringType(), FloatType()])),
                'Json[(Int, String, Float)]',
            ),
            (Json[(int, ...)], JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            (
                Json[(int, str, float, ...)],
                JsonType(JsonType.TypeSchema([IntType(), StringType()], variadic_type=FloatType())),
                'Json[(Int, String, Float, ...)]',
            ),
            (
                Json[{'a': int, 'b': str, 'c': [int], 'd': (int, str, ...), 'e': {'x': int, 'y': TypedDict1}}],
                JsonType(
                    JsonType.TypeSchema(
                        {
                            'a': IntType(),
                            'b': StringType(),
                            'c': JsonType(JsonType.TypeSchema([], variadic_type=IntType())),
                            'd': JsonType(JsonType.TypeSchema([IntType()], variadic_type=StringType())),
                            'e': JsonType(
                                JsonType.TypeSchema(
                                    {
                                        'x': IntType(),
                                        'y': JsonType(
                                            JsonType.TypeSchema(
                                                {
                                                    'a': StringType(),
                                                    'b': IntType(nullable=True),
                                                    'c': ArrayType((None,), dtype=FloatType()),
                                                }
                                            )
                                        ),
                                    }
                                )
                            ),
                        }
                    )
                ),
                "Json[{'a': Int, 'b': String, 'c': Json[(Int, ...)], 'd': Json[(Int, String, ...)], "
                "'e': Json[{'x': Int, 'y': Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}]}]}]",
            ),
            (Image[100, 200], ImageType(width=100, height=200, mode=None, nullable=False), 'Image[(100, 200)]'),
            (Image[100, None], ImageType(width=100, height=None, mode=None, nullable=False), 'Image[(100, None)]'),
            (Image[None, 200], ImageType(width=None, height=200, mode=None, nullable=False), 'Image[(None, 200)]'),
            (
                Image[(100, 200), 'RGB'],
                ImageType(width=100, height=200, mode='RGB', nullable=False),
                "Image[(100, 200), 'RGB']",
            ),
            (Image['RGB'], ImageType(height=None, width=None, mode='RGB', nullable=False), "Image['RGB']"),
            (Literal['a', 'b', 'c'], StringType(nullable=False), 'String'),
            (Literal[1, 2, 3], IntType(nullable=False), 'Int'),
            (Literal[1, 2.0, 3], FloatType(nullable=False), 'Float'),
            (Literal['a', 'b', None], StringType(nullable=True), 'String'),  # noqa: PYI061
        )
        for x, pxt_type, string in test_cases:
            py_type: Any = x  # We need to explicitly type this as Any to avoid a silly mypy subscripting bug
            print(py_type)
            non_nullable_pxt_type = pxt_type.copy(nullable=False)
            nullable_pxt_type = pxt_type.copy(nullable=True)
            assert non_nullable_pxt_type.matches(pxt_type)  # sanity check behavior of matches() and copy()
            assert nullable_pxt_type.matches(pxt_type)

            # a bare type is non-nullable; nullability must be spelled out explicitly
            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(Optional[py_type]) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type]) == nullable_pxt_type  # noqa: RUF036
            assert ColumnType.from_python_type(py_type | None) == nullable_pxt_type
            assert ColumnType.from_python_type(None | py_type) == nullable_pxt_type  # noqa: RUF036

            # `Required[T]` is deprecated, but still resolves to a non-nullable `T`
            with pytest.warns(excs.PixeltableDeprecationWarning, match=r'`Required\[T\]` is deprecated'):
                assert ColumnType.from_python_type(Required[py_type]) == non_nullable_pxt_type

            assert str(non_nullable_pxt_type) == string
            assert str(nullable_pxt_type) == f'{string} | None'

    def test_supertype(self, init_env: None) -> None:
        test_cases = [
            (IntType(), FloatType(), FloatType()),
            (BoolType(), IntType(), IntType()),
            (BoolType(), FloatType(), FloatType()),
            (
                ImageType(height=100, width=200, mode='RGB'),
                ImageType(height=100, width=200, mode='RGB'),
                ImageType(height=100, width=200, mode='RGB'),
            ),
            (
                ImageType(height=100, width=200, mode='RGB'),
                ImageType(height=100, width=200, mode='RGBA'),
                ImageType(height=100, width=200, mode=None),
            ),
            (
                ImageType(height=100, width=200, mode='RGB'),
                ImageType(height=100, width=300, mode='RGB'),
                ImageType(height=100, width=None, mode='RGB'),
            ),
            (
                ImageType(height=100, width=200, mode='RGB'),
                ImageType(height=300, width=200, mode='RGB'),
                ImageType(height=None, width=200, mode='RGB'),
            ),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=300, width=400, mode='RGBA'), ImageType()),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(), ImageType()),
            (JsonType(), JsonType(), JsonType()),
            (IntType(), StringType(), JsonType()),
            (IntType(), JsonType(), JsonType()),
            (JsonType(), IntType(), JsonType()),
            (TimestampType(), IntType(), None),
            (DateType(), StringType(), None),
            (
                JsonType(JsonType.TypeSchema({'a': StringType(), 'c': ArrayType((3,), dtype=FLOAT32)})),
                JsonType(JsonType.TypeSchema({'b': IntType(), 'c': ArrayType((5,), dtype=FLOAT32)})),
                # dict schemas are open, so the supertype declares only the keys in both, as their supertype
                JsonType(JsonType.TypeSchema({'c': ArrayType((None,), dtype=FLOAT32)})),
            ),
            (
                # a key with no common supertype is left undeclared
                JsonType(JsonType.TypeSchema({'a': StringType(), 'b': IntType()})),
                JsonType(JsonType.TypeSchema({'a': AudioType(), 'b': IntType()})),
                JsonType(JsonType.TypeSchema({'b': IntType()})),
            ),
            (
                # a key is required in the supertype only if both require it
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()})),
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}, optional_keys=frozenset(('b',)))),
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}, optional_keys=frozenset(('b',)))),
            ),
            (
                # nested dict schemas combine the same way
                JsonType(
                    JsonType.TypeSchema(
                        {'x': JsonType(JsonType.TypeSchema({'a': IntType(), 'b': IntType()})), 'y': IntType()}
                    )
                ),
                JsonType(JsonType.TypeSchema({'x': JsonType(JsonType.TypeSchema({'a': IntType(), 'c': IntType()}))})),
                JsonType(JsonType.TypeSchema({'x': JsonType(JsonType.TypeSchema({'a': IntType()}))})),
            ),
            (
                # closed schemas: a key that only one declares is kept as optional, since the other's values don't
                # have it; the supertype of two closed schemas is closed
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}, closed=True)),
                JsonType(JsonType.TypeSchema({'a': IntType(), 'c': FloatType()}, closed=True)),
                JsonType(
                    JsonType.TypeSchema(
                        {'a': IntType(), 'b': StringType(), 'c': FloatType()},
                        optional_keys=frozenset(('b', 'c')),
                        closed=True,
                    )
                ),
            ),
            (
                # a closed and an open schema: only the open one's keys are kept, and the supertype is open
                JsonType(JsonType.TypeSchema({'a': IntType(), 'c': FloatType()}, closed=True)),
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()})),
                JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}, optional_keys=frozenset(('b',)))),
            ),
            (
                # a key with no common supertype can be left undeclared only by an open supertype
                JsonType(JsonType.TypeSchema({'a': StringType(), 'b': IntType()}, closed=True)),
                JsonType(JsonType.TypeSchema({'a': AudioType(), 'b': IntType()}, closed=True)),
                JsonType(JsonType.TypeSchema({'b': IntType()})),
            ),
            (
                # a dict schema and a list schema -> default JsonType()
                JsonType(JsonType.TypeSchema({'a': StringType()})),
                JsonType(JsonType.TypeSchema([StringType()])),
                JsonType(),
            ),
            (
                # tuples with same length -> tuple of supertype of each position
                JsonType(JsonType.TypeSchema([StringType(), IntType()])),
                JsonType(JsonType.TypeSchema([StringType(), StringType()])),
                JsonType(JsonType.TypeSchema([StringType(), JsonType()])),
            ),
            (
                # tuples with different lengths -> varadic type if allowed
                JsonType(JsonType.TypeSchema([StringType(), IntType()])),
                JsonType(JsonType.TypeSchema([StringType(), StringType(), FloatType(), BoolType()])),
                JsonType(JsonType.TypeSchema([StringType(), JsonType()], variadic_type=FloatType())),
            ),
            (
                # tuples with different lengths and no allowable variadic type -> default JsonType()
                JsonType(JsonType.TypeSchema([StringType(), IntType()])),
                JsonType(JsonType.TypeSchema([StringType(), StringType(), FloatType(), AudioType()])),
                JsonType(),
            ),
            (
                # merge existing variadic type with fixed element from longer tuple
                JsonType(JsonType.TypeSchema([StringType(), IntType()], variadic_type=ArrayType((3,), dtype=FLOAT32))),
                JsonType(JsonType.TypeSchema([StringType(), StringType(), ArrayType((5,), dtype=FLOAT32)])),
                JsonType(
                    JsonType.TypeSchema([StringType(), JsonType()], variadic_type=ArrayType((None,), dtype=FLOAT32))
                ),
            ),
            (
                # both tuples have variadic types
                JsonType(JsonType.TypeSchema([StringType(), IntType()], variadic_type=ArrayType((3,), dtype=FLOAT32))),
                JsonType(
                    JsonType.TypeSchema([StringType(), StringType()], variadic_type=ArrayType((5,), dtype=FLOAT32))
                ),
                JsonType(
                    JsonType.TypeSchema([StringType(), JsonType()], variadic_type=ArrayType((None,), dtype=FLOAT32))
                ),
            ),
        ]
        for i, (t1, t2, expected) in enumerate(test_cases):
            for n1 in [True, False]:
                for n2 in [True, False]:
                    try:
                        t1n = t1.copy(nullable=n1)
                        t2n = t2.copy(nullable=n2)
                        expectedn = None if expected is None else expected.copy(nullable=(n1 or n2))
                        assert t1n.supertype(t2n) == expectedn, (t1n, t2n)
                        assert t2n.supertype(t1n) == expectedn, (t1n, t2n)
                    except Exception as e:
                        print(t1n)
                        print(t2n)
                        print(expectedn)
                        print(t1n.supertype(t2n))
                        raise type(e)(f'Failed test case {i} with n1={n1}, n2={n2}') from e

    def test_json_literal_inference(self, init_env: None) -> None:
        def d(type_spec: dict[str, ColumnType], optional_keys: tuple[str, ...] = ()) -> JsonType:
            return JsonType(JsonType.TypeSchema(type_spec, optional_keys=frozenset(optional_keys), closed=True))

        # literal types have closed schemas, so their common supertype keeps every key, optional where only some
        # literals have it
        test_cases: list[tuple[list[Any], ColumnType]] = [
            ([{'a': 1}, {'b': 'x'}], d({'a': IntType(), 'b': StringType()}, ('a', 'b'))),
            (
                [{'a': 1, 'c': [1.0]}, {'a': 2}],
                d({'a': IntType(), 'c': JsonType(JsonType.TypeSchema([FloatType()]))}, ('c',)),
            ),
            # nested dicts, in a dict or in a list
            ([{'x': {'a': 1}}, {'x': {'b': 1}}], d({'x': d({'a': IntType(), 'b': IntType()}, ('a', 'b'))})),
            (
                [[{'a': 1}], [{'b': 1}]],
                JsonType(JsonType.TypeSchema([d({'a': IntType(), 'b': IntType()}, ('a', 'b'))])),
            ),
            # values of different scalar types combine to Json
            ([{'a': 1}, {'a': 'x'}], d({'a': JsonType()})),
        ]
        for vals, expected in test_cases:
            assert ColumnType.infer_common_literal_type(vals) == expected, vals

    def test_json_is_supertype_of(self, init_env: None) -> None:
        def d(type_spec: dict[str, ColumnType], optional_keys: tuple[str, ...] = (), closed: bool = False) -> JsonType:
            return JsonType(JsonType.TypeSchema(type_spec, optional_keys=frozenset(optional_keys), closed=closed))

        # (t1, t2, whether t1 is a supertype of t2)
        test_cases = [
            # a schema with fewer keys is a supertype (width subtyping), but not the other way around
            (d({'a': IntType()}), d({'a': IntType(), 'b': StringType()}), True),
            (d({'a': IntType(), 'b': StringType()}), d({'a': IntType()}), False),
            (d({}), d({'a': IntType()}), True),
            # an optional key must still be declared by the subtype, since an undeclared key could have any type
            (d({'a': IntType(), 'b': StringType()}, ('b',)), d({'a': IntType()}), False),
            # a required key may be required or optional in the supertype, but not the other way around
            (d({'a': IntType(), 'b': StringType()}, ('b',)), d({'a': IntType(), 'b': StringType()}), True),
            (d({'a': IntType(), 'b': StringType()}), d({'a': IntType(), 'b': StringType()}, ('b',)), False),
            # value types are covariant
            (d({'a': FloatType()}), d({'a': IntType(), 'b': IntType()}), True),
            (d({'a': IntType()}), d({'a': FloatType()}), False),
            (d({'a': IntType(nullable=True)}), d({'a': IntType()}), True),
            (d({'a': IntType()}), d({'a': IntType(nullable=True)}), False),
            # nested dict schemas follow the same rules
            (d({'x': d({'a': IntType()})}), d({'x': d({'a': IntType(), 'b': IntType()}), 'y': IntType()}), True),
            (d({'x': d({'a': IntType(), 'b': IntType()})}), d({'x': d({'a': IntType()})}), False),
            (JsonType(), d({'a': IntType()}), True),
            (d({'a': IntType()}), JsonType(), False),
            (d({}), JsonType(JsonType.TypeSchema([IntType()])), False),
            # a closed schema, which allows no undeclared keys, may omit an optional key of an open supertype
            (d({'a': IntType(), 'b': StringType()}, ('b',)), d({'a': IntType()}, closed=True), True),
            (d({'a': IntType(), 'b': StringType()}), d({'a': IntType()}, closed=True), False),
            (d({'a': IntType()}), d({'a': IntType(), 'c': IntType()}, closed=True), True),
            (d({'x': d({'a': IntType(), 'b': IntType()}, ('b',))}), d({'x': d({'a': IntType()}, closed=True)}), True),
            # a closed schema is a supertype only of closed schemas that declare a subset of its keys
            (d({'a': IntType()}, closed=True), d({'a': IntType()}), False),
            (d({'a': IntType(), 'b': StringType()}, ('b',), closed=True), d({'a': IntType()}, closed=True), True),
            (d({'a': IntType()}, closed=True), d({'a': IntType(), 'c': IntType()}, closed=True), False),
        ]
        for i, (t1, t2, expected) in enumerate(test_cases):
            assert t1.is_supertype_of(t2) == expected, (i, t1, t2)
            assert t1.is_supertype_of(t2, ignore_nullable=True) == expected, (i, t1, t2)

    def test_json_closed_literals(self, init_env: None) -> None:
        # the type of a dict literal has closed schemas, nested ones included
        lit = exprs.Expr.from_object({'a': 1, 'b': [{'c': 'x'}, 2]})
        assert isinstance(lit, exprs.Literal) and isinstance(lit.col_type, JsonType)
        closed_repr = "Json[{'a': Int, 'b': Json[(Json[{'c': String}, closed=True], Int)]}, closed=True]"
        assert repr(lit.col_type) == closed_repr
        assert repr(lit.col_type.opened()) == "Json[{'a': Int, 'b': Json[(Json[{'c': String}], Int)]}]"

        # closedness is not stored; a loaded literal infers it again
        d = lit.as_dict()
        assert 'closed' not in json.dumps(d)
        assert repr(exprs.Expr.from_dict(d).col_type) == closed_repr

        # a literal given an explicit type keeps it
        open_type = JsonType(JsonType.TypeSchema({'a': IntType()}))
        assert exprs.Literal({'a': 1, 'z': 2}, col_type=open_type).col_type == open_type

    def test_json_validate_literal(self, init_env: None) -> None:
        t = JsonType(
            JsonType.TypeSchema(
                {'a': IntType(), 'b': StringType(), 'x': JsonType(JsonType.TypeSchema({'c': IntType()}))},
                optional_keys=frozenset(('b',)),
            )
        )
        # keys the schema doesn't declare are allowed, at any level
        t.validate_literal({'a': 1, 'x': {'c': 2}, 'z': 'extra'})
        t.validate_literal({'a': 1, 'b': 'y', 'x': {'c': 2, 'z': [None]}})
        # declared keys are still checked
        with pytest.raises(TypeError, match="Missing required key: 'a'"):
            t.validate_literal({'x': {'c': 2}, 'z': 'extra'})
        with pytest.raises(TypeError, match="Missing required key: 'c'"):
            t.validate_literal({'a': 1, 'x': {'z': 2}})
        with pytest.raises(TypeError, match='Expected string, got int'):
            t.validate_literal({'a': 1, 'b': 2, 'x': {'c': 2}})

        # a closed schema allows no undeclared keys
        closed = JsonType(JsonType.TypeSchema({'a': IntType()}, closed=True))
        closed.validate_literal({'a': 1})
        with pytest.raises(TypeError, match="Unexpected key: 'z'"):
            closed.validate_literal({'a': 1, 'z': 'extra'})

    def test_to_json_schema(self, init_env: None) -> None:
        test_cases: list[tuple[ColumnType, dict]] = [
            (StringType(nullable=False), {'type': 'string'}),
            (IntType(nullable=True), {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}),
            (
                JsonType(JsonType.TypeSchema([StringType(), BoolType()])),
                {'type': 'array', 'prefixItems': [{'type': 'string'}, {'type': 'boolean'}], 'items': False},
            ),
            (
                JsonType(JsonType.TypeSchema([StringType(), BoolType()], variadic_type=FloatType())),
                {
                    'type': 'array',
                    'prefixItems': [{'type': 'string'}, {'type': 'boolean'}],
                    'items': {'type': 'number'},
                },
            ),
            (
                JsonType(JsonType.TypeSchema({'a': StringType(), 'b': IntType()}, optional_keys=frozenset(('a',)))),
                {
                    'type': 'object',
                    'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}},
                    'additionalProperties': False,
                    'required': ['b'],
                },
            ),
        ]
        for col_type, expected_schema in test_cases:
            assert col_type.to_json_schema() == expected_schema, col_type

    def test_json_type_errors(self, init_env: None) -> None:
        invalid_type_args = (
            (int, r"Invalid type schema: type argument does not represent a valid JSON type: <class 'int'>"),
            (list[int, str], r'Invalid type schema: `list` or `Sequence` must have at most one type argument'),
            (list[set], r"Python type found in type argument is not a valid Pixeltable type: <class 'set'>"),
            (tuple[int, str, set], r"Python type found in type argument is not a valid Pixeltable type: <class 'set'>"),
            ([int, str], r'Invalid type schema: expected a single-item list; got a list of length 2'),
            ([set], r"Invalid type schema: type argument contains an invalid Pixeltable type: <class 'set'>"),
            ((int, str, set), r"Invalid type schema: type argument contains an invalid Pixeltable type: <class 'set'>"),
            ((int, set, ...), r"Invalid type schema: type argument contains an invalid Pixeltable type: <class 'set'>"),
            ((int, str, ..., float), r'Invalid type schema: `...` allowed only in last position'),
            ((int, str, ..., ...), r'Invalid type schema: `...` allowed only in last position'),
            ((...,), r'Invalid type schema: tuple with only `...` is not allowed'),
            (
                {'a': int, 'b': set},
                r"Invalid type schema: type argument contains an invalid Pixeltable type: <class 'set'>",
            ),
        )
        for subscript, message in invalid_type_args:
            with pytest.raises((excs.RequestError, ValueError), match=message):
                Json[subscript]

    def test_type_errors(self, init_env: None) -> None:
        from .module_with_future_annotations import make_unresolvable_typed_dict

        with pytest.raises(excs.RequestError, match='Bare `Required` is not a valid type'):
            ColumnType.from_python_type(Required)
        with pxt_raises(
            pxt.ErrorCode.INVALID_TYPE,
            match=r"Cannot resolve the field types of TypedDict `Unresolvable`: name 'LocalFloat' is not defined",
        ):
            ColumnType.from_python_type(make_unresolvable_typed_dict())

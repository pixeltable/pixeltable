# mypy: disable-error-code="misc"

import datetime
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import PIL.Image
import pydantic

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
            ({'a': 1, 'b': '2'}, JsonType(JsonType.TypeSchema({'a': IntType(), 'b': StringType()}))),
            (['3', 4], JsonType(JsonType.TypeSchema([StringType(), IntType()]))),
        ]
        for val, expected_type in test_cases:
            assert ColumnType.infer_literal_type(val) == expected_type, val

    def test_from_python_type(self, init_env: None) -> None:
        # Test cases: map of python_type to expected (pxt_type, str(pxt_type))
        test_cases: dict[Any, tuple[ColumnType, str]] = {
            # Builtin and standard types
            str: (StringType(nullable=False), 'String'),
            int: (IntType(nullable=False), 'Int'),
            float: (FloatType(nullable=False), 'Float'),
            bool: (BoolType(nullable=False), 'Bool'),
            datetime.datetime: (TimestampType(nullable=False), 'Timestamp'),
            datetime.date: (DateType(nullable=False), 'Date'),
            uuid.UUID: (UUIDType(nullable=False), 'UUID'),
            bytes: (BinaryType(nullable=False), 'Binary'),
            list: (JsonType(nullable=False), 'Json'),
            dict: (JsonType(nullable=False), 'Json'),
            list[int]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=False), 'Json[(Int, ...)]'),
            list[dict[str, int]]: (
                JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False),
                'Json[(Json, ...)]',
            ),
            dict[int, str]: (JsonType(nullable=False), 'Json'),
            dict[dict[str, int], list[int]]: (JsonType(nullable=False), 'Json'),
            List: (JsonType(nullable=False), 'Json'),
            Dict: (JsonType(nullable=False), 'Json'),
            List[int]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=False), 'Json[(Int, ...)]'),
            List[Dict[str, int]]: (
                JsonType(JsonType.TypeSchema([], variadic_type=JsonType()), nullable=False),
                'Json[(Json, ...)]',
            ),
            Dict[int, str]: (JsonType(nullable=False), 'Json'),
            PIL.Image.Image: (ImageType(nullable=False), 'Image'),
            # Pixeltable types
            String: (StringType(nullable=False), 'String'),
            Int: (IntType(nullable=False), 'Int'),
            Float: (FloatType(nullable=False), 'Float'),
            Bool: (BoolType(nullable=False), 'Bool'),
            Timestamp: (TimestampType(nullable=False), 'Timestamp'),
            Date: (DateType(nullable=False), 'Date'),
            UUID: (UUIDType(nullable=False), 'UUID'),
            Binary: (BinaryType(nullable=False), 'Binary'),
            Array: (ArrayType(nullable=False), 'Array'),
            Json: (JsonType(nullable=False), 'Json'),
            Image: (ImageType(height=None, width=None, mode=None, nullable=False), 'Image'),
            Video: (VideoType(nullable=False), 'Video'),
            Audio: (AudioType(nullable=False), 'Audio'),
            Document: (DocumentType(nullable=False), 'Document'),
            # Pixeltable types with specialized parameters
            Array[Int]: (ArrayType(dtype=IntType(), nullable=False), 'Array[int64]'),
            Array[(None,), Int]: (ArrayType((None,), dtype=IntType(), nullable=False), 'Array[(None,), int64]'),
            Array[(5,), Bool]: (ArrayType((5,), dtype=BoolType(), nullable=False), 'Array[(5,), bool]'),
            Array[(5, None, 3), Float]: (
                ArrayType((5, None, 3), dtype=FloatType(), nullable=False),
                'Array[(5, None, 3), float32]',
            ),
            # Json list or tuple
            Json[list[int]]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            Json[tuple[int, str]]: (JsonType(JsonType.TypeSchema([IntType(), StringType()])), 'Json[(Int, String)]'),
            Json[tuple[int, ...]]: (JsonType(JsonType.TypeSchema([], variadic_type=StringType())), 'Json[(Int, ...)]'),
            Json[List[int]]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            Json[Tuple[int, str]]: (JsonType(JsonType.TypeSchema([IntType(), StringType()])), 'Json[(Int, String)]'),
            # Json TypedDict
            Json[TypedDict1]: (
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(nullable=True), 'c': ArrayType((None,), dtype=FloatType())}
                    )
                ),
                "Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}]",
            ),
            Json[TypedDict2]: (
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(nullable=True), 'c': ArrayType((None,), dtype=FloatType())},
                        optional_keys=['a', 'b', 'c'],
                    )
                ),
                "Json[{'a': String, 'b': Int | None, 'c': Array[(None,), float32]}, optional_keys=['a', 'b', 'c']]",
            ),
            Json[TypedDict3]: (
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
            Json[Model1]: (
                JsonType(
                    JsonType.TypeSchema(
                        {
                            'a': StringType(),
                            'b': JsonType(JsonType.TypeSchema([], variadic_type=IntType()), nullable=True),
                            'c': JsonType(JsonType.TypeSchema([], variadic_type=StringType())),
                            'd': IntType(nullable=True),
                        },
                        optional_keys=['d'],
                    )
                ),
                "Json[{'a': String, 'b': Json[(Int, ...)] | None, 'c': Json[(String, ...)], 'd': Int | None}, "
                "optional_keys=['d']]",
            ),
            # Json "convenience structures"
            Json[[int]]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            Json[(int,)]: (JsonType(JsonType.TypeSchema([IntType()])), 'Json[(Int,)]'),
            Json[(int, str, float)]: (
                JsonType(JsonType.TypeSchema([IntType(), StringType(), FloatType()])),
                'Json[(Int, String, Float)]',
            ),
            Json[(int, ...)]: (JsonType(JsonType.TypeSchema([], variadic_type=IntType())), 'Json[(Int, ...)]'),
            Json[(int, str, float, ...)]: (
                JsonType(JsonType.TypeSchema([IntType(), StringType()], variadic_type=FloatType())),
                'Json[(Int, String, Float, ...)]',
            ),
            Json[{'a': int, 'b': str, 'c': [int], 'd': (int, str, ...), 'e': {'x': int, 'y': TypedDict1}}]: (
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
            Image[100, 200]: (ImageType(width=100, height=200, mode=None, nullable=False), 'Image[(100, 200)]'),
            Image[100, None]: (ImageType(width=100, height=None, mode=None, nullable=False), 'Image[(100, None)]'),
            Image[None, 200]: (ImageType(width=None, height=200, mode=None, nullable=False), 'Image[(None, 200)]'),
            Image[(100, 200), 'RGB']: (
                ImageType(width=100, height=200, mode='RGB', nullable=False),
                "Image[(100, 200), 'RGB']",
            ),
            Image['RGB']: (ImageType(height=None, width=None, mode='RGB', nullable=False), "Image['RGB']"),
            Literal['a', 'b', 'c']: (StringType(nullable=False), 'String'),
            Literal[1, 2, 3]: (IntType(nullable=False), 'Int'),
            Literal[1, 2.0, 3]: (FloatType(nullable=False), 'Float'),
            Literal['a', 'b', None]: (StringType(nullable=True), 'String'),  # noqa: PYI061
        }
        for py_type, (pxt_type, string) in test_cases.items():
            print(py_type)
            non_nullable_pxt_type = pxt_type.copy(nullable=False)
            nullable_pxt_type = pxt_type.copy(nullable=True)

            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(Required[py_type]) == non_nullable_pxt_type  # type: ignore[valid-type]
            assert ColumnType.from_python_type(Optional[py_type]) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type]) == nullable_pxt_type  # noqa: RUF036
            assert ColumnType.from_python_type(py_type | None) == nullable_pxt_type
            assert ColumnType.from_python_type(None | py_type) == nullable_pxt_type  # noqa: RUF036

            assert ColumnType.from_python_type(py_type, nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Required[py_type], nullable_default=True) == non_nullable_pxt_type  # type: ignore[valid-type]
            assert ColumnType.from_python_type(Optional[py_type], nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type], nullable_default=True) == nullable_pxt_type  # noqa: RUF036
            assert ColumnType.from_python_type(py_type | None, nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(None | py_type, nullable_default=True) == nullable_pxt_type  # noqa: RUF036

            assert str(non_nullable_pxt_type) == string
            assert str(nullable_pxt_type) == f'{string} | None'
            assert non_nullable_pxt_type._to_str(as_schema=True) == f'Required[{string}]'
            assert nullable_pxt_type._to_str(as_schema=True) == string

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
                # keys in one but not the other become optional keys in the supertype;
                # keys in both become their supertype
                JsonType(
                    JsonType.TypeSchema(
                        {'a': StringType(), 'b': IntType(), 'c': ArrayType((None,), dtype=FLOAT32)},
                        optional_keys=['a', 'b'],
                    )
                ),
            ),
            (
                # conflicting keys -> default JsonType()
                JsonType(JsonType.TypeSchema({'a': StringType()})),
                JsonType(JsonType.TypeSchema({'a': AudioType()})),
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
                JsonType(JsonType.TypeSchema({'a': StringType(), 'b': IntType()}, optional_keys=['a'])),
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

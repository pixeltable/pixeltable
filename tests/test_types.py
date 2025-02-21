import datetime
from typing import Any, Dict, List, Optional

import jsonschema.exceptions
import numpy as np
import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable.type_system import *

from .utils import skip_test_if_not_installed


class TestTypes:
    json_schema_1 = {
        'properties': {
            'a': {'type': 'string'},  # required in 1 and 2
            'b': {'type': 'integer'},  # required in 1, optional in 2
            'c': {'type': 'number'},  # required in 2, optional in 1
            'd': {'type': 'boolean'},  # optional in 1 and 2
            'e': {'type': 'string'},  # required in 1, absent from 2
            'g': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},  # nullable in 1, non-nullable in 2
            'h': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},  # type conflict (string in 1, int in 2)
        },
        'required': ['a', 'b'],
    }

    json_schema_2 = {
        'properties': {
            'a': {'type': 'string'},
            'b': {'type': 'integer'},
            'c': {'type': 'number'},
            'd': {'type': 'boolean'},
            'f': {'type': 'string'},  # required in 2, absent from 1
            'g': {'type': 'string'},
            'h': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
        },
        'required': ['a', 'c'],
    }

    json_schema_12 = {  # supertype of 1 + 2
        'type': 'object',
        'properties': {
            'a': {'type': 'string'},
            'b': {'type': 'integer'},
            'c': {'type': 'number'},
            'd': {'type': 'boolean'},
            'e': {'type': 'string'},
            'f': {'type': 'string'},
            'g': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
            'h': {},
        },
        'required': ['a'],
    }

    bad_json_schema = {'type': 'junk'}

    def test_infer(self, init_env) -> None:
        test_cases: list[tuple[Any, ColumnType]] = [
            ('a', StringType()),
            (1, IntType()),
            (1.0, FloatType()),
            (True, BoolType()),
            (datetime.datetime.now(), TimestampType()),
            (PIL.Image.new('RGB', (100, 100)), ImageType(height=100, width=100, mode='RGB')),
            (np.ndarray((1, 2, 3), dtype=np.int64), ArrayType((1, 2, 3), dtype=IntType())),
            ({'a': 1, 'b': '2'}, pxt.JsonType()),
            (['3', 4], pxt.JsonType()),
        ]
        for val, expected_type in test_cases:
            assert ColumnType.infer_literal_type(val) == expected_type, val

    def test_from_python_type(self, init_env) -> None:
        # Test cases: map of python_type to expected (pxt_type, str(pxt_type))
        test_cases: dict[type, tuple[ColumnType, str]] = {
            # Builtin and standard types
            str: (StringType(nullable=False), 'String'),
            int: (IntType(nullable=False), 'Int'),
            float: (FloatType(nullable=False), 'Float'),
            bool: (BoolType(nullable=False), 'Bool'),
            datetime.datetime: (TimestampType(nullable=False), 'Timestamp'),
            list: (JsonType(nullable=False), 'Json'),
            dict: (JsonType(nullable=False), 'Json'),
            list[int]: (JsonType(nullable=False), 'Json'),
            list[dict[str, int]]: (JsonType(nullable=False), 'Json'),
            dict[int, str]: (JsonType(nullable=False), 'Json'),
            dict[dict[str, int], list[int]]: (JsonType(nullable=False), 'Json'),
            List: (JsonType(nullable=False), 'Json'),
            Dict: (JsonType(nullable=False), 'Json'),
            List[int]: (JsonType(nullable=False), 'Json'),
            List[Dict[str, int]]: (JsonType(nullable=False), 'Json'),
            Dict[int, str]: (JsonType(nullable=False), 'Json'),
            PIL.Image.Image: (ImageType(nullable=False), 'Image'),
            # Pixeltable types
            String: (StringType(nullable=False), 'String'),
            Int: (IntType(nullable=False), 'Int'),
            Float: (FloatType(nullable=False), 'Float'),
            Bool: (BoolType(nullable=False), 'Bool'),
            Timestamp: (TimestampType(nullable=False), 'Timestamp'),
            Array: (ArrayType(nullable=False), 'Array'),
            Json: (JsonType(nullable=False), 'Json'),
            Image: (ImageType(height=None, width=None, mode=None, nullable=False), 'Image'),
            Video: (VideoType(nullable=False), 'Video'),
            Audio: (AudioType(nullable=False), 'Audio'),
            Document: (DocumentType(nullable=False), 'Document'),
            # Pixeltable types with specialized parameters
            Array[Int]: (ArrayType(dtype=IntType(), nullable=False), 'Array[Int]'),  # type: ignore[misc]
            Array[(None,), Int]: (ArrayType((None,), dtype=IntType(), nullable=False), 'Array[(None,), Int]'),  # type: ignore[misc]
            Array[(5,), Bool]: (ArrayType((5,), dtype=BoolType(), nullable=False), 'Array[(5,), Bool]'),  # type: ignore[misc]
            Array[(5, None, 3), Float]: (  # type: ignore[misc]
                ArrayType((5, None, 3), dtype=FloatType(), nullable=False),
                'Array[(5, None, 3), Float]',
            ),
            Image[(100, 200)]: (ImageType(width=100, height=200, mode=None, nullable=False), 'Image[(100, 200)]'),  # type: ignore[misc]
            Image[(100, None)]: (ImageType(width=100, height=None, mode=None, nullable=False), 'Image[(100, None)]'),  # type: ignore[misc]
            Image[(None, 200)]: (ImageType(width=None, height=200, mode=None, nullable=False), 'Image[(None, 200)]'),  # type: ignore[misc]
            Image[(100, 200), 'RGB']: (  # type: ignore[misc]
                ImageType(width=100, height=200, mode='RGB', nullable=False),
                "Image[(100, 200), 'RGB']",
            ),
            Image['RGB']: (ImageType(height=None, width=None, mode='RGB', nullable=False), "Image['RGB']"),  # type: ignore[misc]
            Literal['a', 'b', 'c']: (StringType(nullable=False), "String"),
            Literal[1, 2, 3]: (IntType(nullable=False), "Int"),
            Literal[1, 2.0, 3]: (FloatType(nullable=False), "Float"),
            Literal['a', 'b', None]: (StringType(nullable=True), "String"),
        }
        for py_type, (pxt_type, string) in test_cases.items():
            print(py_type)
            non_nullable_pxt_type = pxt_type.copy(nullable=False)
            nullable_pxt_type = pxt_type.copy(nullable=True)

            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(Required[py_type]) == non_nullable_pxt_type  # type: ignore[valid-type]
            assert ColumnType.from_python_type(Optional[py_type]) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type]) == nullable_pxt_type

            assert ColumnType.from_python_type(py_type, nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Required[py_type], nullable_default=True) == non_nullable_pxt_type  # type: ignore[valid-type]
            assert ColumnType.from_python_type(Optional[py_type], nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type], nullable_default=True) == nullable_pxt_type

            assert str(non_nullable_pxt_type) == string
            assert str(nullable_pxt_type) == f'Optional[{string}]'
            assert non_nullable_pxt_type._to_str(as_schema=True) == f'Required[{string}]'
            assert nullable_pxt_type._to_str(as_schema=True) == string

    def test_supertype(self, init_env) -> None:
        test_cases = [
            (IntType(), FloatType(), FloatType()),
            (BoolType(), IntType(), IntType()),
            (BoolType(), FloatType(), FloatType()),
            (IntType(), StringType(), None),
            (
                ArrayType((1, 2, 3), dtype=IntType()),
                ArrayType((3, 2, 1), dtype=IntType()),
                ArrayType((None, 2, None), dtype=IntType()),
            ),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((1, 2), dtype=IntType()), ArrayType(dtype=IntType())),
            (
                ArrayType((1, 2, 3), dtype=IntType()),
                ArrayType((3, 2, 1), dtype=FloatType()),
                ArrayType((None, 2, None), dtype=FloatType()),
            ),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((3, 2, 1), dtype=StringType()), ArrayType()),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((1, 2), dtype=StringType()), ArrayType()),
            (ArrayType(), IntType(), None),
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
            (JsonType(json_schema=self.json_schema_1), JsonType(), JsonType()),
            (
                JsonType(json_schema=self.json_schema_1),
                JsonType(json_schema=self.json_schema_2),
                JsonType(json_schema=self.json_schema_12),
            ),
            (JsonType(), IntType(), None),
        ]
        for t1, t2, expected in test_cases:
            for n1 in [True, False]:
                for n2 in [True, False]:
                    t1n = t1.copy(nullable=n1)
                    t2n = t2.copy(nullable=n2)
                    expectedn = None if expected is None else expected.copy(nullable=(n1 or n2))
                    assert t1n.supertype(t2n) == expectedn, (t1n, t2n)
                    assert t2n.supertype(t1n) == expectedn, (t1n, t2n)

    def test_json_schemas(self, init_env) -> None:
        skip_test_if_not_installed('pydantic')
        import pydantic

        class SampleModel(pydantic.BaseModel):
            a: str
            b: int
            c: Optional[bool]

        json_type = ColumnType.from_python_type(Json[SampleModel.model_json_schema()])  # type: ignore[misc]
        assert isinstance(json_type, JsonType)
        assert str(json_type) == 'Json[SampleModel]'

        with pytest.raises(jsonschema.exceptions.SchemaError) as exc_info:
            Json[self.bad_json_schema]  # type: ignore[misc]
        assert "'junk' is not valid under any of the given schemas" in str(exc_info.value)

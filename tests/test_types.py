import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import PIL.Image

import pixeltable as pxt
from pixeltable.type_system import *


class TestTypes:
    def test_infer(self) -> None:
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

    def test_serialize(self, init_env) -> None:
        type_vals = [
            InvalidType(), StringType(), IntType(), BoolType(), TimestampType(),
            ImageType(height=100, width=200, mode='RGB'),
            JsonType({
                'a': StringType(), 'b': IntType(), 'c': FloatType(), 'd': BoolType(), 'e': TimestampType(),
                'f': ImageType(height=100, width=200, mode='RGB'),
                'g': JsonType({'f1': StringType(), 'f2': IntType()}),
                'h': ArrayType((224, 224, 3), dtype=IntType()),
            }),
            ArrayType((224, 224, 3), dtype=IntType()),
        ]

        for t in type_vals:
            t_serialized = t.serialize()
            t_deserialized = ColumnType.deserialize(t_serialized)
            assert t == t_deserialized

    def test_from_python_type(self) -> None:
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
            Image: (ImageType(height=None, width=None, mode=None, nullable=False), 'Image'),
            Json: (JsonType(nullable=False), 'Json'),
            Video: (VideoType(nullable=False), 'Video'),
            Audio: (AudioType(nullable=False), 'Audio'),
            Document: (DocumentType(nullable=False), 'Document'),

            # Pixeltable types with specialized parameters
            Array[(None,), Int]: (ArrayType((None,), dtype=IntType(), nullable=False), 'Array[(None,), Int]'),
            Array[(5, None, 3), Float]: (ArrayType((5, None, 3), dtype=FloatType(), nullable=False), 'Array[(5, None, 3), Float]'),
            Image[(100, 200)]: (ImageType(width=100, height=200, mode=None, nullable=False), 'Image[(100, 200)]'),
            Image[(100, None)]: (ImageType(width=100, height=None, mode=None, nullable=False), 'Image[(100, None)]'),
            Image[(None, 200)]: (ImageType(width=None, height=200, mode=None, nullable=False), 'Image[(None, 200)]'),
            Image[(100, 200), 'RGB']: (ImageType(width=100, height=200, mode='RGB', nullable=False), "Image[(100, 200), 'RGB']"),
            Image['RGB']: (ImageType(height=None, width=None, mode='RGB', nullable=False), "Image['RGB']"),
        }
        for py_type, (pxt_type, string) in test_cases.items():
            assert not pxt_type.nullable
            nullable_pxt_type = pxt_type.copy(nullable=True)

            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(Required[py_type]) == pxt_type
            assert ColumnType.from_python_type(Optional[py_type]) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type]) == nullable_pxt_type

            assert ColumnType.from_python_type(py_type, nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Required[py_type], nullable_default=True) == pxt_type
            assert ColumnType.from_python_type(Optional[py_type], nullable_default=True) == nullable_pxt_type
            assert ColumnType.from_python_type(Union[None, py_type], nullable_default=True) == nullable_pxt_type

            assert str(pxt_type) == string
            assert str(nullable_pxt_type) == f'Optional[{string}]'
            assert pxt_type._to_str(as_schema=True) == f'Required[{string}]'
            assert nullable_pxt_type._to_str(as_schema=True) == string

    def test_supertype(self) -> None:
        test_cases = [
            (IntType(), FloatType(), FloatType()),
            (BoolType(), IntType(), IntType()),
            (BoolType(), FloatType(), FloatType()),
            (IntType(), StringType(), None),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((3, 2, 1), dtype=IntType()), ArrayType((None, 2, None), dtype=IntType())),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((1, 2), dtype=IntType()), None),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((3, 2, 1), dtype=FloatType()), ArrayType((None, 2, None), dtype=FloatType())),
            (ArrayType((1, 2, 3), dtype=IntType()), ArrayType((3, 2, 1), dtype=StringType()), None),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=100, width=200, mode='RGB'), ImageType(height=100, width=200, mode='RGB')),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=100, width=200, mode='RGBA'), ImageType(height=100, width=200, mode=None)),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=100, width=300, mode='RGB'), ImageType(height=100, width=None, mode='RGB')),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=300, width=200, mode='RGB'), ImageType(height=None, width=200, mode='RGB')),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(height=300, width=400, mode='RGBA'), ImageType()),
            (ImageType(height=100, width=200, mode='RGB'), ImageType(), ImageType()),
            (JsonType(), JsonType(), JsonType()),
            (JsonType(type_spec={'a': IntType()}), JsonType(), JsonType()),
            (JsonType(type_spec={'a': IntType()}), JsonType(type_spec={'b': StringType()}), JsonType(type_spec={'a': IntType(), 'b': StringType()})),
            (JsonType(type_spec={'a': IntType()}), JsonType(type_spec={'a': StringType()}), JsonType()),
            (JsonType(), IntType(), None),
        ]
        for t1, t2, expected in test_cases:
            for n1 in [True, False]:
                for n2 in [True, False]:
                    t1n = t1.copy(nullable=n1)
                    t2n = t2.copy(nullable=n2)
                    expectedn = None if expected is None else expected.copy(nullable=(n1 or n2))
                    assert t1n.supertype(t2n) == expectedn
                    assert t2n.supertype(t1n) == expectedn

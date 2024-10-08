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
        # Standard/builtin Python types
        test_cases = {
            str: StringType(),
            int: IntType(),
            float: FloatType(),
            bool: BoolType(),
            datetime.datetime: TimestampType(),
            list: JsonType(),
            dict: JsonType(),
            list[int]: JsonType(),
            list[dict[str, int]]: JsonType(),
            dict[int, str]: JsonType(),
            dict[dict[str, int], list[int]]: JsonType(),
            List: JsonType(),
            Dict: JsonType(),
            List[int]: JsonType(),
            List[Dict[str, int]]: JsonType(),
            Dict[int, str]: JsonType(),
            PIL.Image.Image: ImageType(),
        }
        for py_type, pxt_type in test_cases.items():
            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(Optional[py_type]) == pxt_type.copy(nullable=True)
            assert ColumnType.from_python_type(Union[None, py_type]) == pxt_type.copy(nullable=True)

        # Pixeltable types that are nullable by default and accept T[NotNull] syntax
        pxt_test_cases = {
            String: StringType(nullable=True),
            Int: IntType(nullable=True),
            Float: FloatType(nullable=True),
            Bool: BoolType(nullable=True),
            Timestamp: TimestampType(nullable=True),
            Image: ImageType(height=None, width=None, mode=None, nullable=True),
            Json: JsonType(nullable=True),
            Video: VideoType(nullable=True),
            Audio: AudioType(nullable=True),
            Document: DocumentType(nullable=True),
        }
        for py_type, pxt_type in pxt_test_cases.items():
            assert ColumnType.from_python_type(py_type) == pxt_type
            assert ColumnType.from_python_type(py_type[NotNull]) == pxt_type.copy(nullable=False)

        # Pixeltable types with additional specialized parameters
        parameterized_pxt_test_cases = {
            Array[(None,), Int]: ArrayType((None,), dtype=IntType(), nullable=True),
            Array[(None,), Int, NotNull]: ArrayType((None,), dtype=IntType(), nullable=False),
            Array[NotNull, (None,), Int]: ArrayType((None,), dtype=IntType(), nullable=False),
            Array[(5, None, 3), Float]: ArrayType((5, None, 3), dtype=FloatType(), nullable=True),
            Image[(100, 200)]: ImageType(width=100, height=200, mode=None, nullable=True),
            Image[(100, 200), NotNull]: ImageType(width=100, height=200, mode=None, nullable=False),
            Image[(100, 200), 'RGB']: ImageType(width=100, height=200, mode='RGB', nullable=True),
            Image['RGB', NotNull, (100, 200)]: ImageType(width=100, height=200, mode='RGB', nullable=False),
            Image['RGB']: ImageType(height=None, width=None, mode='RGB', nullable=True),
        }
        for py_type, pxt_type in parameterized_pxt_test_cases.items():
            assert ColumnType.from_python_type(py_type) == pxt_type

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

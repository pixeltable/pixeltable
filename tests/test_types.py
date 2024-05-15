import datetime
from copy import copy
from typing import List, Dict, Optional

from pixeltable.type_system import \
    ColumnType, StringType, IntType, BoolType, ImageType, InvalidType, FloatType, TimestampType, JsonType, ArrayType


class TestTypes:
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
        test_cases = {
            str: StringType(),
            int: IntType(),
            float: FloatType(),
            bool: BoolType(),
            datetime.date: TimestampType(),
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
            Dict[int, str]: JsonType()
        }
        for py_type, pxt_type in test_cases.items():
            assert ColumnType.from_python_type(py_type) == pxt_type
            opt_pxt_type = copy(pxt_type)
            opt_pxt_type.nullable = True
            assert ColumnType.from_python_type(Optional[py_type]) == opt_pxt_type

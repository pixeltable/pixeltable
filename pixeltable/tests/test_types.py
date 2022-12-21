import pytest
import pandas as pd

import pixeltable as pt
from pixeltable.type_system import \
    ColumnType, StringType, IntType, BoolType, ImageType, InvalidType, FloatType, TimestampType, JsonType, ArrayType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog


class TestTypes:
    def test_serialize(self, init_db: None) -> None:
        type_vals = [
            InvalidType(), StringType(), IntType(), BoolType(), TimestampType(),
            ImageType(height=100, width=200, mode=ImageType.Mode.RGB),
            JsonType({
                'a': StringType(), 'b': IntType(), 'c': FloatType(), 'd': BoolType(), 'e': TimestampType(),
                'f': ImageType(height=100, width=200, mode=ImageType.Mode.RGB),
                'g': JsonType({'f1': StringType(), 'f2': IntType()}),
                'h': ArrayType((224, 224, 3), dtype=ColumnType.Type.INT),
            }),
            ArrayType((224, 224, 3), dtype=ColumnType.Type.INT),
        ]

        for t in type_vals:
            t_serialized = t.serialize()
            t_deserialized = ColumnType.deserialize(t_serialized)
            assert t == t_deserialized

import pytest

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import StringType, IntType, FloatType, TimestampType


class TestTest:
    def test_one(self, test_db: pt.Db) -> None:
        test_db.create_dir('dir1')

    def test_two(self, test_db: pt.Db) -> None:
        c1 = catalog.Column('c1', StringType(), nullable=False)
        c2 = catalog.Column('c2', IntType(), nullable=False)
        c3 = catalog.Column('c3', FloatType(), nullable=False)
        c4 = catalog.Column('c4', TimestampType(), nullable=False)
        schema = [c1, c2, c3, c4]
        _ = test_db.create_table('test', schema)

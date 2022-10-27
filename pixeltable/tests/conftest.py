from typing import List
import pytest
import sqlalchemy as sql

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ColumnType

@pytest.fixture(scope='function')
def test_db() -> None:
    engine = sql.create_engine('sqlite:///:memory:', echo=True)
    pt.store.set_engine(engine)
    pt.store.init_db()

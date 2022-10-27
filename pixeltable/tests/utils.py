import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ColumnType


def make_tbl(db: pt.Db, name: str = 'test', col_names: List[str] = ['c1']) -> pt.MutableTable:
    schema: List[catalog.Column] = []
    for i, col_name in enumerate(col_names):
        schema.append(catalog.Column(f'{col_name}', ColumnType(i % len(ColumnType))))
    return db.create_table(name, schema)

def create_test_data(t: catalog.Table, num_rows: int = 10) -> pd.DataFrame:
    data: Dict[str, Any] = {}
    for col in t.columns():
        col_data: Any = None
        if col.col_type == ColumnType.STRING:
            col_data = ['test string'] * num_rows
        if col.col_type == ColumnType.INT:
            col_data = np.random.randint(0, 100, size=num_rows)
        if col.col_type == ColumnType.FLOAT:
            col_data = np.random.random(size=num_rows) * 100
        if col.col_type == ColumnType.BOOL:
            col_data = np.random.randint(0, 2, size=num_rows)
        if col.col_type == ColumnType.TIMESTAMP:
            col_data = datetime.datetime.now()
        # TODO: implement this
        assert col.col_type != ColumnType.IMAGE
        assert col.col_type != ColumnType.DICT
        assert col.col_type != ColumnType.VECTOR
        data[col.name] = col_data
    return pd.DataFrame(data=data)
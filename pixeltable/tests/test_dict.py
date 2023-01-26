import sqlalchemy as sql
import pytest

from pixeltable import catalog
from pixeltable.type_system import IntType, JsonType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file

from pixeltable.env import Env

class TestExprs:
    def test_basic(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        res = t[t.c6].show()
        stmt = sql.select(sql.func.jsonb_path_query(t.cols_by_name['c6'].sa_col, '$.detections[*].bounding_box'))
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['iscrowd']).where(sql.cast(t.cols_by_name['c2'].sa_col['iscrowd'], sql.Integer) == 0)
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['iscrowd']).where(t.cols_by_name['c2'].sa_col['supercategory'] == '"furniture"')
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['bounding_box', 0]).where(t.cols_by_name['c2'].sa_col['supercategory'] == '"furniture"')
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['bounding_box', 0]).where(t.cols_by_name['c2'].sa_col['supercategory'].astext == 'furniture')
        with Env.get().engine.connect() as conn:
            result = conn.execute(stmt)
            for row in result:
                print(row)
        print(res)


import sqlalchemy as sql
import pytest

from pixeltable import catalog
from pixeltable.type_system import IntType, DictType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file

from pixeltable import env

class TestExprs:
    def test_basic(self, test_db: catalog.Db) -> None:
        db = test_db
        c1 = catalog.Column('c1', IntType(), nullable=False)
        c2 = catalog.Column('c2', DictType(), nullable=False)
        schema = [c1, c2]
        t = db.create_table('test', schema)
        pd_df = create_table_data(t)
        print(pd_df)
        t.insert_pandas(pd_df)
        _ = t.count()
        res = t[t.c2].show()
        stmt = sql.select(sql.func.jsonb_path_query(t.cols_by_name['c2'].sa_col, '$.detections[*].bounding_box'))
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['iscrowd']).where(sql.cast(t.cols_by_name['c2'].sa_col['iscrowd'], sql.Integer) == 0)
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['iscrowd']).where(t.cols_by_name['c2'].sa_col['supercategory'] == '"furniture"')
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['bounding_box', 0]).where(t.cols_by_name['c2'].sa_col['supercategory'] == '"furniture"')
        #stmt = sql.select(t.cols_by_name['c2'].sa_col['bounding_box', 0]).where(t.cols_by_name['c2'].sa_col['supercategory'].astext == 'furniture')
        with env.get_engine().connect() as conn:
            result = conn.execute(stmt)
            for row in result:
                print(row)
        print(res)


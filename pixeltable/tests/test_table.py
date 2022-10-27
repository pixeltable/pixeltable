import pytest

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import ColumnType
from pixeltable.tests.utils import make_tbl, create_test_data


class TestTable:
    def test_create(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        db.create_dir('dir1')
        c1 = catalog.Column('c1', ColumnType.STRING, nullable=False)
        c2 = catalog.Column('c2', ColumnType.INT, nullable=False)
        c3 = catalog.Column('c3', ColumnType.FLOAT, nullable=False)
        c4 = catalog.Column('c4', ColumnType.TIMESTAMP, nullable=False)
        schema = [c1, c2, c3, c4]
        _ = db.create_table('test', schema)
        _ = db.create_table('dir1.test', schema)

        with pytest.raises(exc.BadFormatError):
            _ = db.create_table('1test', schema)
        with pytest.raises(exc.DuplicateNameError):
            _ = db.create_table('test', schema)
        with pytest.raises(exc.DuplicateNameError):
            _ = db.create_table('test2', [c1, c1])
        with pytest.raises(exc.UnknownEntityError):
            _ = db.create_table('dir2.test2', schema)

        _ = db.list_tables()
        _ = db.list_tables('dir1')

        with pytest.raises(exc.BadFormatError):
            _ = db.list_tables('1dir')
        with pytest.raises(exc.UnknownEntityError):
            _ = db.list_tables('dir2')

        # test loading with new client
        cl2 = pt.Client()
        db = cl2.get_db('test')

        tbl = db.get_table('test')
        assert isinstance(tbl, catalog.MutableTable)
        tbl.add_column(catalog.Column('c5', ColumnType.INT))
        tbl.drop_column('c1')
        tbl.rename_column('c2', 'c17')

        db.rename_table('test', 'test2')

        db.drop_table('test2')
        db.drop_table('dir1.test')

        with pytest.raises(exc.UnknownEntityError):
            db.drop_table('test')
        with pytest.raises(exc.UnknownEntityError):
            db.drop_table('dir1.test2')
        with pytest.raises(exc.BadFormatError):
            db.drop_table('.test2')

    @pytest.mark.dependency()
    def test_insert(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        data1 = create_test_data(t1)
        t1.insert(data1)
        assert len(pt.DataFrame(t1).show().rows) == len(data1)
        data2 = create_test_data(t1)
        t1.insert(data2)
        assert len(pt.DataFrame(t1).show().rows) == len(data1) + len(data2)
        t1.revert()
        assert len(pt.DataFrame(t1).show().rows) == len(data1)
        t1.insert(data2)
        assert len(pt.DataFrame(t1).show().rows) == len(data1) + len(data2)

        # incompatible schema
        t2 = make_tbl(db, 'test2', ['c2', 'c1'])
        t2_data = create_test_data(t2)
        with pytest.raises(exc.InsertError):
            t1.insert(t2_data)

    @pytest.mark.dependency('test_insert')
    def test_query(self, test_db: None) -> None:
        cl = pt.Client()
        db = cl.create_db('test')
        t = make_tbl(db, 'test', ['c1', 'c2'])
        t_data = create_test_data(t)
        t.insert(t_data)
        df = pt.DataFrame(t)
        _ = df.show()

        # test querying existing table
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table('test')
        _  = pt.DataFrame(t2).show()



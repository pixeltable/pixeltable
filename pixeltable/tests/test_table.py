import pytest

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import StringType, IntType, FloatType, TimestampType, ImageType, JsonType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file


class TestTable:
    def test_create(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_dir('dir1')
        c1 = catalog.Column('c1', StringType(), nullable=False)
        c2 = catalog.Column('c2', IntType(), nullable=False)
        c3 = catalog.Column('c3', FloatType(), nullable=False)
        c4 = catalog.Column('c4', TimestampType(), nullable=False)
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
        tbl.add_column(catalog.Column('c5', IntType()))
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

    def test_create_images(self, test_db: catalog.Db) -> None:
        db = test_db
        cols = [
            catalog.Column('img', ImageType(), nullable=False),
            catalog.Column('category', StringType(), nullable=False),
            catalog.Column('split', StringType(), nullable=False),
        ]
        tbl = db.create_table('test', cols)
        df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        # TODO: insert a random subset
        tbl.insert_pandas(df[:20])
        html_str = tbl.show(n=100)._repr_html_()
        print(html_str)
        # TODO: check html_str

    @pytest.mark.dependency(name='test_insert')
    def test_insert(self, test_db: catalog.Db) -> None:
        db = test_db
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        data1 = create_table_data(t1)
        t1.insert_pandas(data1)
        assert t1.count() == len(data1)

        # incompatible schema
        t2 = make_tbl(db, 'test2', ['c2', 'c1'])
        t2_data = create_table_data(t2)
        with pytest.raises(exc.InsertError):
            t1.insert_pandas(t2_data)

    @pytest.mark.dependency(depends=['test_insert'])
    def test_query(self, test_db: catalog.Db) -> None:
        db = test_db
        t = make_tbl(db, 'test', ['c1', 'c2', 'c3', 'c4', 'c5'])
        t_data = create_table_data(t)
        t.insert_pandas(t_data)
        _ = t.show(n=0)

        # test querying existing table
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table('test')
        _  = t2.show(n=0)

    def test_computed_cols(self, test_db: catalog.Db) -> None:
        db = test_db
        c1 = catalog.Column('c1', IntType(), nullable=False)
        c2 = catalog.Column('c2', FloatType(), nullable=False)
        c3 = catalog.Column('c3', JsonType(), nullable=False)
        schema = [c1, c2, c3]
        t = db.create_table('test', schema)
        t.add_column(catalog.Column('c4', value_expr=t.c1 + 1))
        t.add_column(catalog.Column('c5', value_expr=t.c4 + 1))
        t.add_column(catalog.Column('c6', value_expr=t.c1 / t.c2))
        t.add_column(catalog.Column('c7', value_expr=t.c6 * t.c2))
        t.add_column(catalog.Column('c8', value_expr=t.c3.detections['*'].bounding_box))
        assert len(t.c1.col.dependent_cols) == 2
        assert len(t.c2.col.dependent_cols) == 2
        assert len(t.c3.col.dependent_cols) == 1
        assert len(t.c4.col.dependent_cols) == 1
        assert len(t.c5.col.dependent_cols) == 0
        assert len(t.c6.col.dependent_cols) == 1
        assert len(t.c7.col.dependent_cols) == 0
        assert len(t.c8.col.dependent_cols) == 0

        data_df = create_table_data(t, ['c1', 'c2', 'c3'], num_rows=10)
        t.insert_pandas(data_df)
        _ = t.show()

        # not allowed to pass values for computed cols
        with pytest.raises(exc.InsertError):
            data_df2 = create_table_data(t, num_rows=10)
            t.insert_pandas(data_df2)

        # test loading from store
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table('test')
        assert len(t.columns) == len(t2.columns)
        for i in range(len(t.columns)):
            if t.columns[i].value_expr is not None:
                assert t.columns[i].value_expr.equals(t2.columns[i].value_expr)

        # make sure we're still seeing the same data
        tbl_df = t2.show().to_pandas()
        for c in ['c1', 'c2', 'c3']:
            assert tbl_df[c].equals(data_df[c])

        t2.insert_pandas(data_df)
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(exc.OperationalError):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    @pytest.mark.dependency(depends=['test_insert'])
    def test_revert(self, test_db: catalog.Db) -> None:
        db = test_db
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        data1 = create_table_data(t1)
        t1.insert_pandas(data1)
        assert t1.count() == len(data1)
        data2 = create_table_data(t1)
        t1.insert_pandas(data2)
        assert t1.count() == len(data1) + len(data2)
        t1.revert()
        assert t1.count() == len(data1)
        t1.insert_pandas(data2)
        assert t1.count() == len(data1) + len(data2)

    @pytest.mark.dependency(depends=['test_insert'])
    def test_snapshot(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_dir('main')
        tbl = make_tbl(db, 'main.test1', ['c1', 'c2'])
        data1 = create_table_data(tbl)
        tbl.insert_pandas(data1)
        assert tbl.count() == len(data1)

        db.create_snapshot('snap', ['main.test1'])
        snap = db.get_table('snap.test1')
        assert snap.count() == len(data1)

        # adding data to a base table doesn't change the snapshot
        data2 = create_table_data(tbl)
        tbl.insert_pandas(data2)
        assert tbl.count() == len(data1) + len(data2)
        assert snap.count() == len(data1)

        tbl.revert()
        # can't revert a version referenced by a snapshot
        with pytest.raises(exc.OperationalError):
            tbl.revert()

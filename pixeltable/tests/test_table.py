import pandas as pd
import pytest
import math
import numpy as np

import PIL

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file, get_video_files
from pixeltable.functions import make_video, sum
from pixeltable.utils.imgstore import ImageStore


class TestTable:
    # exc for a % 10 == 0
    @pt.function(return_type=FloatType(), param_types=[IntType()])
    def f1(a: int) -> float:
        return a / (a % 10)

    # exception for a == None; this should not get triggered
    @pt.function(return_type=FloatType(), param_types=[FloatType()])
    def f2(a: float) -> float:
        return a + 1

    def test_create(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_dir('dir1')
        c1 = catalog.Column('c1', StringType(), nullable=False)
        c2 = catalog.Column('c2', IntType(), nullable=False)
        c3 = catalog.Column('c3', FloatType(), nullable=False)
        c4 = catalog.Column('c4', TimestampType(), nullable=False)
        schema = [c1, c2, c3, c4]
        tbl = db.create_table('test', schema)
        _ = db.create_table('dir1.test', schema)

        with pytest.raises(exc.Error):
            _ = db.create_table('1test', schema)
        with pytest.raises(exc.Error):
            _ = catalog.Column('1c', StringType())
        with pytest.raises(exc.Error):
            _ = db.create_table('test2', [c1, c1])
        with pytest.raises(exc.Error):
            _ = db.create_table('test', schema)
        with pytest.raises(exc.Error):
            _ = db.create_table('test2', [c1, c1])
        with pytest.raises(exc.Error):
            _ = db.create_table('dir2.test2', schema)

        _ = db.list_tables()
        _ = db.list_tables('dir1')

        with pytest.raises(exc.Error):
            _ = db.list_tables('1dir')
        with pytest.raises(exc.Error):
            _ = db.list_tables('dir2')

        # 'stored' kwarg only applies to computed image columns
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', IntType(), stored=False))
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', ImageType(), stored=False))
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', computed_with=(tbl.c2 + tbl.c3), stored=False))

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

        with pytest.raises(exc.Error):
            db.drop_table('test')
        with pytest.raises(exc.Error):
            db.drop_table('dir1.test2')
        with pytest.raises(exc.Error):
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

    def test_create_video_table(self, test_db: catalog.Db) -> None:
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        tbl = db.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=0)
        # create_table() didn't mess with our 'cols' variable
        assert cols[1].stored == None
        tbl.add_column(catalog.Column('c1', computed_with=tbl.frame.rotate(30), stored=True))
        tbl.add_column(catalog.Column('c2', computed_with=tbl.c1.rotate(40), stored=False))
        tbl.add_column(catalog.Column('c3', computed_with=tbl.c2.rotate(50), stored=True))
        # a non-materialized column that refers to another non-materialized column
        tbl.add_column(catalog.Column('c4', computed_with=tbl.c2.rotate(60), stored=False))

        class WindowFnAggregator:
            def __init__(self):
                pass
            @classmethod
            def make_aggregator(cls) -> 'WindowFnAggregator':
                return cls()
            def update(self) -> None:
                pass
            def value(self) -> int:
                return 1
        window_fn = pt.make_aggregate_function(
            IntType(), [],
            init_fn=WindowFnAggregator.make_aggregator,
            update_fn=WindowFnAggregator.update,
            value_fn=WindowFnAggregator.value,
            requires_order_by=True, allows_window=True)
        # cols computed with window functions are stored by default
        tbl.add_column((catalog.Column('c5', computed_with=window_fn(tbl.frame_idx, group_by=tbl.video))))
        assert tbl.cols_by_name['c5'].is_stored

        # cannot store frame col
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False, stored=True),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        with pytest.raises(exc.Error):
            _ = db.create_table(
                'test', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)

        params = tbl.parameters
        # reload to make sure that metadata gets restored correctly
        cl = pt.Client()
        db = cl.get_db('test')
        tbl = db.get_table('test')
        assert tbl.parameters == params
        tbl.insert_rows([[get_video_files()[0]]], ['video'])
        # * 2: we have four stored img cols
        assert ImageStore.count(tbl.id) == tbl.count() * 2
        html_str = tbl.show(n=100)._repr_html_()

        # revert() clears stored images
        tbl.revert()
        assert ImageStore.count(tbl.id) == 0

        with pytest.raises(exc.Error):
            # can't drop frame col
            tbl.drop_column('frame')
        with pytest.raises(exc.Error):
            # can't drop frame_idx col
            tbl.drop_column('frame_idx')

        # drop() clears stored images and the cache
        tbl.insert_rows([[get_video_files()[0]]], ['video'])
        _ = tbl.show()
        tbl.drop()
        assert ImageStore.count(tbl.id) == 0

        with pytest.raises(exc.Error):
            # missing parameters
            _ = db.create_table(
                'exc', cols, extract_frames_from='video',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.Error):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='frame', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.Error):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame_idx',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.Error):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='frame', extracted_fps=0)
        with pytest.raises(exc.Error):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='breaks', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.Error):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='breaks',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.Error):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='breaks', extracted_fps=0)

    def test_insert(self, test_db: catalog.Db) -> None:
        db = test_db
        t1 = make_tbl(db, 'test1', ['c1', 'c2'])
        data1 = create_table_data(t1)
        t1.insert_pandas(data1)
        assert t1.count() == len(data1)

        # incompatible schema
        t2 = make_tbl(db, 'test2', ['c2', 'c1'])
        t2_data = create_table_data(t2)
        with pytest.raises(exc.Error):
            t1.insert_pandas(t2_data)

        # TODO: test data checks

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
        t.add_column(catalog.Column('c4', computed_with=t.c1 + 1))
        t.add_column(catalog.Column('c5', computed_with=t.c4 + 1))
        t.add_column(catalog.Column('c6', computed_with=t.c1 / t.c2))
        t.add_column(catalog.Column('c7', computed_with=t.c6 * t.c2))
        t.add_column(catalog.Column('c8', computed_with=t.c3.detections['*'].bounding_box))
        t.add_column(catalog.Column('c9', FloatType(), computed_with=lambda c2: math.sqrt(c2)))

        # unstored cols that compute window functions aren't currently supported
        with pytest.raises((exc.Error)):
            t.add_column(catalog.Column('c10', computed_with=sum(t.c1, group_by=t.c1), stored=False))

        # Column.dependent_cols are computed correctly
        assert len(t.c1.col.dependent_cols) == 2
        assert len(t.c2.col.dependent_cols) == 3
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
        with pytest.raises(exc.Error):
            data_df2 = create_table_data(t, ['c1', 'c2', 'c3', 'c4'], num_rows=10)
            t.insert_pandas(data_df2)

        # computed col references non-existent col
        with pytest.raises(exc.Error):
            c1 = catalog.Column('c1', IntType(), nullable=False)
            c2 = catalog.Column('c2', FloatType(), nullable=False)
            c3 = catalog.Column('c3', FloatType(), nullable=False, computed_with=lambda c2: math.sqrt(c2))
            _ = db.create_table('test2', [c1, c3, c2])

        # test loading from store
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table('test')
        assert len(t.columns) == len(t2.columns)
        for i in range(len(t.columns)):
            if t.columns[i].value_expr is not None:
                assert t.columns[i].value_expr.equals(t2.columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert_pandas(data_df)
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(exc.Error):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    def test_computed_col_exceptions(self, test_db: catalog.Db, test_tbl: catalog.Table) -> None:
        db = test_db
        c2 = catalog.Column('c2', IntType(), nullable=False)
        schema = [c2]
        t = db.create_table('test', schema)

        status = t.add_column(catalog.Column('add1', computed_with=self.f2(self.f1(t.c2))))

        data_df = test_tbl[test_tbl.c2].show(0).to_pandas()
        status = t.insert_pandas(data_df)
        _ = str(status)
        assert status.num_excs == 10
        assert 'add1' in status.cols_with_excs
        result_set = t[t.add1.errortype != None][t.add1.errortype, t.add1.errormsg].show(0)
        assert len(result_set) == 10

    def _test_computed_img_cols(self, t: catalog.Table, stores_img_col: bool) -> None:
        data_df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        t.insert_pandas(data_df.loc[0:20, ['img']])
        _ = t.count()
        _ = t.show()
        assert ImageStore.count(t.id) == t.count() * stores_img_col

        # test loading from store
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table(t.name)
        assert len(t.columns) == len(t2.columns)
        for i in range(len(t.columns)):
            if t.columns[i].value_expr is not None:
                assert t.columns[i].value_expr.equals(t2.columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert_pandas(data_df.loc[0:20, ['img']])
        assert ImageStore.count(t2.id) == t2.count() * stores_img_col
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()
        print(tbl_df)

        # revert also removes computed images
        t2.revert()
        assert ImageStore.count(t2.id) == t2.count() * stores_img_col

    def test_computed_img_cols(self, test_db: catalog.Db) -> None:
        db = test_db
        c1 = catalog.Column('img', ImageType(), nullable=False, indexed=True)
        schema = [c1]
        t = db.create_table('test', schema)
        t.add_column(catalog.Column('c2', computed_with=t.img.width))
        # c3 is not stored by default
        t.add_column(catalog.Column('c3', computed_with=t.img.rotate(90)))
        self._test_computed_img_cols(t, stores_img_col=False)

        t = db.create_table('test2', schema)
        # c3 is now stored
        t.add_column(catalog.Column('c3', computed_with=t.img.rotate(90), stored=True))
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t[t.c3.errortype].show(0)

        # computed img col with exceptions
        t = db.create_table('test3', schema)
        @pt.function(return_type=ImageType(), param_types=[ImageType()])
        def f(img: PIL.Image.Image) -> PIL.Image.Image:
            raise RuntimeError
        t.add_column(catalog.Column('c3', computed_with=f(t.img), stored=True))
        data_df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        t.insert_pandas(data_df.loc[0:20, ['img']])
        _ = t[t.c3.errortype].show(0)

    def test_computed_window_fn(self, test_db: catalog.Db, test_tbl: catalog.Table) -> None:
        db = test_db
        t = test_tbl
        # backfill
        t.add_column(catalog.Column('c9', computed_with=sum(t.c2, group_by=t.c4, order_by=t.c3)))

        c2 = catalog.Column('c2', IntType(), nullable=False)
        c3 = catalog.Column('c3', FloatType(), nullable=False)
        c4 = catalog.Column('c4', BoolType(), nullable=False)
        new_t = db.create_table('insert_test', [c2, c3, c4])
        new_t.add_column(catalog.Column('c5', IntType(), computed_with=lambda c2: c2 * c2))
        new_t.add_column(catalog.Column(
            'c6', computed_with=sum(new_t.c5, group_by=new_t.c4, order_by=new_t.c3)))
        data_df = t[t.c2, t.c4, t.c3].show(0).to_pandas()
        new_t.insert_pandas(data_df)
        _ = new_t.show(0)
        print(_)

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

    def test_snapshot(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_dir('main')
        tbl = make_tbl(db, 'main.test1', ['c1', 'c2'])
        data1 = create_table_data(tbl)
        tbl.insert_pandas(data1)
        assert tbl.count() == len(data1)

        db.create_dir('snap')
        db.create_snapshot('snap.test1', 'main.test1')
        snap = db.get_table('snap.test1')
        assert snap.count() == len(data1)

        # adding data to a base table doesn't change the snapshot
        data2 = create_table_data(tbl)
        tbl.insert_pandas(data2)
        assert tbl.count() == len(data1) + len(data2)
        assert snap.count() == len(data1)

        tbl.revert()
        # can't revert a version referenced by a snapshot
        with pytest.raises(exc.Error):
            tbl.revert()

    def test_add_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t.add_column(catalog.Column('add1', pt.IntType(), nullable=False))
        assert len(t.columns) == num_orig_cols + 1

        # make sure this is still true after reloading the metadata
        cl = pt.Client()
        db = cl.get_db('test')
        t = db.get_table(t.name)
        assert len(t.columns) == num_orig_cols + 1

        # revert() works
        t.revert()
        assert len(t.columns) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pt.Client()
        db = cl.get_db('test')
        t = db.get_table(t.name)
        assert len(t.columns) == num_orig_cols

    def test_drop_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns)
        t.drop_column('c1')
        assert len(t.columns) == num_orig_cols - 1

        # make sure this is still true after reloading the metadata
        cl = pt.Client()
        db = cl.get_db('test')
        t = db.get_table(t.name)
        assert len(t.columns) == num_orig_cols - 1

        # revert() works
        t.revert()
        assert len(t.columns) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pt.Client()
        db = cl.get_db('test')
        t = db.get_table(t.name)
        assert len(t.columns) == num_orig_cols

    def test_add_computed_column(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        status = t.add_column(catalog.Column('add1', computed_with=t.c2 + 10, nullable=False))
        assert status.num_excs == 0
        _ = t.show()

        # with exception in SQL
        with pytest.raises(exc.Error):
            t.add_column(catalog.Column('add2', computed_with=(t.c2 - 10) / (t.c3 - 10), nullable=False))

        # with exception in Python for c6.f2 == 10
        status = t.add_column(catalog.Column('add2', computed_with=(t.c6.f2 - 10) / (t.c6.f2 - 10), nullable=False))
        assert status.num_excs == 1
        result = t[t.add2.errortype != None][t.c6.f2, t.add2, t.add2.errortype, t.add2.errormsg].show()
        assert len(result) == 1

        # test case: exceptions in dependencies prevent execution of dependent exprs
        status = t.add_column(catalog.Column('add3', computed_with=self.f2(self.f1(t.c2))))
        assert status.num_excs == 10
        result = t[t.add3.errortype != None][t.c2, t.add3, t.add3.errortype, t.add3.errormsg].show()
        assert len(result) == 10

    def test_describe(self, test_tbl: catalog.Table) -> None:
        t = test_tbl
        fn = lambda c2: np.full((3, 4), c2)
        t.add_column(
            catalog.Column('computed1', col_type=ArrayType((3, 4), dtype=IntType()), computed_with=fn))
        _ = t.describe()
        print(_)

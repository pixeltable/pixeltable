import pandas as pd
import pytest
import math

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file, get_video_files, sum_uda
from pixeltable.functions import make_video
from pixeltable import utils


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
        with pytest.raises(exc.BadFormatError):
            _ = catalog.Column('1c', StringType())
        with pytest.raises(exc.DuplicateNameError):
            _ = db.create_table('test2', [c1, c1])
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

    def test_create_video(self, test_db: catalog.Db) -> None:
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        tbl = db.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=0)
        params = tbl.parameters
        # reload to make sure that metadata gets restored correctly
        cl = pt.Client()
        db = cl.get_db('test')
        tbl = db.get_table('test')
        assert tbl.parameters == params
        tbl.insert_rows([[get_video_files()[0]]], ['video'])
        html_str = tbl.show(n=100)._repr_html_()
        # TODO: check html_str
        _ = tbl[make_video(tbl.frame_idx, tbl.frame)].group_by(tbl.video).show()

        with pytest.raises(exc.Error):
            # can't drop frame col
            tbl.drop_column('frame')
        with pytest.raises(exc.Error):
            # can't drop frame_idx col
            tbl.drop_column('frame_idx')
        with pytest.raises(exc.BadFormatError):
            # missing parameters
            _ = db.create_table(
                'exc', cols, extract_frames_from='video',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='frame', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame_idx',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # wrong column type
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='frame', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='breaks', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='breaks',
                extracted_frame_idx_col='frame_idx', extracted_fps=0)
        with pytest.raises(exc.BadFormatError):
            # unknown column
            _ = db.create_table(
                'exc', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='breaks', extracted_fps=0)

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
        t.add_column(catalog.Column('c4', computed_with=t.c1 + 1))
        t.add_column(catalog.Column('c5', computed_with=t.c4 + 1))
        t.add_column(catalog.Column('c6', computed_with=t.c1 / t.c2))
        t.add_column(catalog.Column('c7', computed_with=t.c6 * t.c2))
        t.add_column(catalog.Column('c8', computed_with=t.c3.detections['*'].bounding_box))
        t.add_column(catalog.Column('c9', FloatType(), computed_with=lambda c2: math.sqrt(c2)))

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
        with pytest.raises(exc.InsertError):
            data_df2 = create_table_data(t, num_rows=10)
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

    def test_computed_img_cols(self, test_db: catalog.Db) -> None:
        db = test_db
        c1 = catalog.Column('img', ImageType(), nullable=False, indexed=True)
        schema = [c1]
        t = db.create_table('test', schema)
        t.add_column(catalog.Column('c2', computed_with=t.img.width))
        t.add_column(catalog.Column('c3', computed_with=t.img.rotate(90)))

        data_df = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        t.insert_pandas(data_df.loc[0:20, ['img']])
        _ = t.show()
        assert utils.computed_img_count(tbl_id=t.id) == t.count()

        # test loading from store
        cl2 = pt.Client()
        db2 = cl2.get_db('test')
        t2 = db2.get_table('test')
        assert len(t.columns) == len(t2.columns)
        for i in range(len(t.columns)):
            if t.columns[i].value_expr is not None:
                assert t.columns[i].value_expr.equals(t2.columns[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert_pandas(data_df.loc[0:20, ['img']])
        assert utils.computed_img_count(tbl_id=t.id) == t2.count()
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()
        print(tbl_df)

        # revert also removes computed images
        t2.revert()
        assert utils.computed_img_count() == t2.count()

    def test_computed_window_fn(self, test_db: catalog.Db, test_tbl: catalog.Table) -> None:
        db = test_db
        t = test_tbl
        # backfill
        t.add_column(catalog.Column('c9', computed_with=sum_uda(t.c2).window(partition_by=t.c4, order_by=t.c3)))

        c2 = catalog.Column('c2', IntType(), nullable=False)
        c3 = catalog.Column('c3', FloatType(), nullable=False)
        c4 = catalog.Column('c4', BoolType(), nullable=False)
        new_t = db.create_table('insert_test', [c2, c3, c4])
        new_t.add_column(catalog.Column('c5', IntType(), computed_with=lambda c2: c2 * c2))
        new_t.add_column(catalog.Column(
            'c6', computed_with=sum_uda(new_t.c5).window(partition_by=new_t.c4, order_by=new_t.c3)))
        data_df = t[t.c2, t.c4, t.c3].show(0).to_pandas()
        new_t.insert_pandas(data_df)
        _ = new_t.show(0)
        print(_)

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

    def test_add_column(self, test_db: catalog.Db) -> None:
        db = test_db
        t = make_tbl(db, 'test', ['c1', 'c2'])
        data1 = create_table_data(t)
        t.insert_pandas(data1)
        assert t.count() == len(data1)
        t.add_column(catalog.Column('c3', computed_with=t.c2 + 10, nullable=False))
        _ = t.show()
        print(_)

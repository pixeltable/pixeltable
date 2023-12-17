import pytest
import math
import numpy as np
import pandas as pd
import datetime
import random
import os

import PIL
import cv2

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import make_tbl, create_table_data, read_data_file, get_video_files, assert_resultset_eq
from pixeltable.functions import make_video, sum
from pixeltable.utils.imgstore import ImageStore
from pixeltable.utils.filecache import FileCache
from pixeltable.iterators import FrameIterator


class TestTable:
    # exc for a % 10 == 0
    @pt.udf(return_type=FloatType(), param_types=[IntType()])
    def f1(a: int) -> float:
        return a / (a % 10)

    # exception for a == None; this should not get triggered
    @pt.udf(return_type=FloatType(), param_types=[FloatType()])
    def f2(a: float) -> float:
        return a + 1

    def test_create(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_dir('dir1')
        c1 = catalog.Column('c1', StringType(nullable=False))
        c2 = catalog.Column('c2', IntType(nullable=False))
        c3 = catalog.Column('c3', FloatType(nullable=False))
        c4 = catalog.Column('c4', TimestampType(nullable=False))
        schema = [c1, c2, c3, c4]
        tbl = cl.create_table('test', schema)
        _ = cl.create_table('dir1.test', schema)

        with pytest.raises(exc.Error):
            _ = cl.create_table('1test', schema)
        with pytest.raises(exc.Error):
            _ = catalog.Column('1c', StringType())
        with pytest.raises(exc.Error):
            _ = cl.create_table('test2', [c1, c1])
        with pytest.raises(exc.Error):
            _ = cl.create_table('test', schema)
        with pytest.raises(exc.Error):
            _ = cl.create_table('test2', [c1, c1])
        with pytest.raises(exc.Error):
            _ = cl.create_table('dir2.test2', schema)

        _ = cl.list_tables()
        _ = cl.list_tables('dir1')

        with pytest.raises(exc.Error):
            _ = cl.list_tables('1dir')
        with pytest.raises(exc.Error):
            _ = cl.list_tables('dir2')

        # 'stored' kwarg only applies to computed image columns
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', IntType(), stored=False))
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', ImageType(), stored=False))
        with pytest.raises(exc.Error):
            tbl.add_column(catalog.Column('c5', computed_with=(tbl.c2 + tbl.c3), stored=False))

        # test loading with new client
        cl = pt.Client()

        tbl = cl.get_table('test')
        assert isinstance(tbl, catalog.InsertableTable)
        tbl.add_column(catalog.Column('c5', IntType()))
        tbl.drop_column('c1')
        tbl.rename_column('c2', 'c17')

        cl.move('test', 'test2')

        cl.drop_table('test2')
        cl.drop_table('dir1.test')

        with pytest.raises(exc.Error):
            cl.drop_table('test')
        with pytest.raises(exc.Error):
            cl.drop_table('dir1.test2')
        with pytest.raises(exc.Error):
            cl.drop_table('.test2')

    def test_create_image_table(self, test_client: pt.Client) -> None:
        cl = test_client
        cols = [
            catalog.Column('img', ImageType(nullable=False)),
            catalog.Column('category', StringType(nullable=False)),
            catalog.Column('split', StringType(nullable=False)),
            catalog.Column('img_literal', ImageType(nullable=False)),
        ]
        tbl = cl.create_table('test', cols)

        rows, col_names = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        sample_rows = random.sample(rows, 20)

        # add literal image data and column
        for r in rows:
            with open(r[0], 'rb') as f:
                r.append(f.read())

        col_names.append('img_literal')
        tbl.insert(sample_rows, columns=col_names)

        # compare img and img_literal
        # TODO: make tbl.select(tbl.img == tbl.img_literal) work
        tdf = tbl.select(tbl.img, tbl.img_literal).show()
        pdf = tdf.to_pandas()
        for tup in pdf.itertuples():
            assert tup.img == tup.img_literal

    def test_insert_validate_image(self, test_client: pt.Client) -> None:
        cols = [
            catalog.Column('img', ImageType(nullable=True)),
            catalog.Column('is_bad_image', BoolType(nullable=False))
        ]
        tbl = test_client.create_table('test', cols)
        rows, col_names = read_data_file('imagenette2-160', 'manifest_bad.csv', ['img'])
        assert len(rows) > 0

        ## Mode 1: Validation error on bad input (default)
        with pytest.raises(exc.DataValidationError) as exc_info:
            tbl.insert(rows, columns=col_names)
        assert 'bad_image.JPEG' in str(exc_info.value)

        ## Mode 2: ignore_errors=True, store error information in table
        tbl.insert(rows, columns=col_names, ignore_errors=True)
        bad_rows = tbl.where(tbl.is_bad_image == True)
        good_rows = tbl.where(tbl.is_bad_image == False)

        # check that we have the right number of bad and good rows
        assert bad_rows.count() == 1
        assert good_rows.count() == 3

        # check error type is set correctly
        assert tbl.where((tbl.is_bad_image == True) & (tbl.img.errortype == None)).count() == 0
        assert tbl.where((tbl.is_bad_image == False) & (tbl.img.errortype == None)).count() == 3

        # check fileurl is set for valid images, and check no file url is set for bad images
        # NB(orm) found no intuitive way to express this without getting data out
        # (see https://www.notion.so/API-pain-points-60841135411c4c67bb422dc0179dc08a )
        assert (tbl.where((tbl.is_bad_image == False)).select(tbl.img.fileurl).show()
                .to_pandas()['img_fileurl'].isnull().sum()) == 0
        
        assert (tbl.where((tbl.is_bad_image == True)).select(tbl.img.fileurl).show()
                .to_pandas()['img_fileurl'].isnull().sum()) == 1
        
    def test_insert_validate_video(self, test_client: pt.Client) -> None:
        cols = [
            catalog.Column('video', VideoType(nullable=True)),
            catalog.Column('is_bad_video', BoolType(nullable=False))
        ]
        tbl = test_client.create_table('test_video_tab', cols)
        files = get_video_files(include_bad_video=True)
        rows = [[f, f.endswith('bad_video.mp4')] for f in files]
        col_names = ['video', 'is_bad_video']
        assert len(rows) > 0

        ## Mode 1: Validation error on bad input (default)
        with pytest.raises(exc.DataValidationError) as exc_info:
            tbl.insert(rows, columns=col_names)
        assert 'bad_video.mp4' in str(exc_info.value)

        ## Mode 2: ignore_errors=True, store error information in table
        tbl.insert(rows, columns=col_names, ignore_errors=True)
        bad_rows = tbl.where(tbl.is_bad_video == True)
        good_rows = tbl.where(tbl.is_bad_video == False)

        # check that we have the right number of bad and good rows
        assert bad_rows.count() == 1
        assert good_rows.count() == 3

        # check error type is set correctly
        assert tbl.where((tbl.is_bad_video == True) & (tbl.video.errortype == None)).count() == 0
        assert tbl.where((tbl.is_bad_video == False) & (tbl.video.errortype == None)).count() == 3

        # check fileurl is set for valid images, and check no file url is set for bad images
        # NB(orm) found no intuitive way to express this without getting data out
        # (see https://www.notion.so/API-pain-points-60841135411c4c67bb422dc0179dc08a )
        assert (tbl.where((tbl.is_bad_video == False)).select(tbl.video.fileurl).show()
                .to_pandas()['video_fileurl'].isnull().sum()) == 0
        
        assert (tbl.where((tbl.is_bad_video == True)).select(tbl.video.fileurl).show()
                .to_pandas()['video_fileurl'].isnull().sum()) == 1

    def test_create_s3_image_table(self, test_client: pt.Client) -> None:
        cl = test_client
        tbl = cl.create_table('test', [catalog.Column('img', ImageType(nullable=False))])
        # add computed column to make sure that external files are cached locally during insert
        tbl.add_column(catalog.Column('rotated', computed_with=tbl.img.rotate(30), stored=True))
        urls = [
            's3://open-images-dataset/validation/3c02ca9ec9b2b77b.jpg',
            's3://open-images-dataset/validation/3c13e0015b6c3bcf.jpg',
            's3://open-images-dataset/validation/3ba5380490084697.jpg',
            's3://open-images-dataset/validation/3afeb4b34f90c0cf.jpg',
            's3://open-images-dataset/validation/3b07a2c0d5c0c789.jpg',
        ]

        tbl.insert([[url] for url in urls], columns=['img'])
        # check that we populated the cache
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 5
        assert cache_stats.num_hits == 0
        assert FileCache.get().num_files() == 5
        assert FileCache.get().num_files(tbl.id) == 5
        assert FileCache.get().avg_file_size() > 0

        # query: we read from the cache
        _ = tbl.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 10
        assert cache_stats.num_hits == 5

        # after clearing the cache, we need to re-fetch the files
        FileCache.get().clear()
        _ = tbl.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 5
        assert cache_stats.num_hits == 0

        # start with fresh client and FileCache instance to test FileCache initialization with pre-existing files
        cl = pt.Client()
        # is there a better way to do this?
        FileCache._instance = None
        t = cl.get_table('test')
        _ = tbl.show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == 5
        assert cache_stats.num_hits == 5

    def test_video_url(self, test_client: pt.Client) -> None:
        cl = test_client
        cols = [
            catalog.Column('payload', IntType(nullable=False)),
            catalog.Column('video', VideoType(nullable=False)),
        ]
        tbl = cl.create_table('test', cols)
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        tbl.insert([[1, url]], columns=['payload', 'video'])
        row = tbl.select(tbl.video.fileurl, tbl.video.localpath).collect()[0]
        assert row['video_fileurl'] == url
        # row[1] contains valid path to an mp4 file
        local_path = row['video_localpath']
        assert os.path.exists(local_path) and os.path.isfile(local_path)
        cap = cv2.VideoCapture(local_path)
        # TODO: this isn't sufficient to determine that this is actually a video, rather than an image
        assert cap.isOpened()
        cap.release()

    def test_create_video_table(self, test_client: pt.Client) -> None:
        cl = test_client
        tbl = cl.create_table(
            'test_tbl',
            [catalog.Column('payload', IntType(nullable=False)), catalog.Column('video', VideoType(nullable=True))])
        args = {'video': tbl.video, 'fps': 0}
        view = cl.create_view('test_view', tbl, iterator_class=FrameIterator, iterator_args=args)
        view.add_column(catalog.Column('c1', computed_with=view.frame.rotate(30), stored=True))
        view.add_column(catalog.Column('c2', computed_with=view.c1.rotate(40), stored=False))
        view.add_column(catalog.Column('c3', computed_with=view.c2.rotate(50), stored=True))
        # a non-materialized column that refers to another non-materialized column
        view.add_column(catalog.Column('c4', computed_with=view.c2.rotate(60), stored=False))

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
        view.add_column((catalog.Column('c5', computed_with=window_fn(view.frame_idx, group_by=view.video))))
        assert view.cols_by_name['c5'].is_stored

        params = view.parameters
        # reload to make sure that metadata gets restored correctly
        cl = pt.Client()
        tbl = cl.get_table('test_tbl')
        view = cl.get_table('test_view')
        assert view.parameters == params
        # we're inserting only a single row and the video column is not in position 0
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        status = tbl.insert([[1, url]], columns=['payload', 'video'])
        assert status.num_excs == 0
        # * 2: we have 2 stored img cols
        assert ImageStore.count(view.id) == view.count() * 2
        # also insert a local file
        tbl.insert([[1, get_video_files()[0]]], columns=['payload', 'video'])
        assert ImageStore.count(view.id) == view.count() * 2

        # TODO: test inserting Nulls
        #status = tbl.insert([[1, None]], columns=['payload', 'video'])
        #assert status.num_excs == 0

        # revert() clears stored images
        tbl.revert()
        tbl.revert()
        assert ImageStore.count(view.id) == 0

        with pytest.raises(exc.Error):
            # can't drop frame col
            view.drop_column('frame')
        with pytest.raises(exc.Error):
            # can't drop frame_idx col
            view.drop_column('frame_idx')

        # drop() clears stored images and the cache
        tbl.insert([[1, get_video_files()[0]]], columns=['payload', 'video'])
        with pytest.raises(exc.Error) as exc_info:
            tbl.drop()
        assert '1 view' in str(exc_info.value)
        view.drop()
        tbl.drop()
        assert ImageStore.count(view.id) == 0

    def test_insert(self, test_client: pt.Client) -> None:
        cl = test_client
        c1 = catalog.Column('c1', StringType(nullable=False))
        c2 = catalog.Column('c2', IntType(nullable=False))
        c3 = catalog.Column('c3', FloatType(nullable=False))
        c4 = catalog.Column('c4', BoolType(nullable=False))
        c5 = catalog.Column('c5', ArrayType((2, 3), dtype=IntType(), nullable=False))
        c6 = catalog.Column('c6', JsonType(nullable=False))
        c7 = catalog.Column('c7', ImageType(nullable=False))
        c8 = catalog.Column('c8', VideoType(nullable=False))
        cols = [c1, c2, c3, c4, c5, c6, c7, c8]
        t = cl.create_table('test1', cols)
        rows = create_table_data(t)
        status = t.insert(rows)
        assert status.num_rows == len(rows)
        assert status.num_excs == 0

        # empty input
        with pytest.raises(exc.Error) as exc_info:
            t.insert([])
        assert 'empty' in str(exc_info.value)

        # missing column
        with pytest.raises(exc.Error) as exc_info:
            t.insert([r[:-1] for r in rows], columns=[c.name for c in cols[:-1]])
        assert 'Missing' in str(exc_info.value)

        # incompatible schema
        for col, row_pos in zip(cols, [1, 2, 4, 4, 5, 6, 1, 1]):
            with pytest.raises(exc.Error) as exc_info:
                cl.drop_table('test1', ignore_errors=True)
                t = cl.create_table('test1', [col])
                t.insert([[r[row_pos]] for r in rows])
            assert 'Expected' in str(exc_info.value)

        # rows not list of lists
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c5])
            t.insert([np.ndarray((3, 2))])
        assert 'lists' in str(exc_info.value)

        # columns not list of column names
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c1])
            t.insert([['test']], columns=[t.c1])
        assert 'column names' in str(exc_info.value)

        # columns not list of column names
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c1])
            t.insert([['test']], columns='c1')
        assert 'column names' in str(exc_info.value)

        # unknown column
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c1])
            t.insert([['test']], columns=['does_not_exist'])
        assert 'unknown' in str(exc_info.value).lower()

        # bad null value
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c1])
            t.insert([[None]], columns=['c1'])
        assert 'contains None' in str(exc_info.value)

        # bad array literal
        with pytest.raises(exc.Error) as exc_info:
            cl.drop_table('test1', ignore_errors=True)
            t = cl.create_table('test1', [c5])
            t.insert([[np.ndarray((3, 2))]])
        assert 'Expected' in str(exc_info.value)

    def test_query(self, test_client: pt.Client) -> None:
        cl = test_client
        col_names = ['c1', 'c2', 'c3', 'c4', 'c5']
        t = make_tbl(cl, 'test', col_names)
        rows = create_table_data(t)
        t.insert(rows, columns=col_names)
        _ = t.show(n=0)

        # test querying existing table
        cl = pt.Client()
        t2 = cl.get_table('test')
        _  = t2.show(n=0)

    def test_update(self, test_tbl: pt.MutableTable, indexed_img_tbl: pt.MutableTable) -> None:
        t = test_tbl
        # update every type with a literal
        test_cases = [
            ('c1', 'new string'),
            # TODO: ('c1n', None),
            ('c3', -1.0),
            ('c4', True),
            ('c5', datetime.datetime.now()),
            ('c6', [{'x': 1, 'y': 2}]),
        ]
        count = t.count()
        for col_name, literal in test_cases:
            status = t.update({col_name: literal}, where=t.c3 < 10.0, cascade=False)
            assert status.num_rows == 10
            assert status.updated_cols == [f'{t.name}.{col_name}']
            assert t.count() == count
            t.revert()

        # exchange two columns
        t.add_column(catalog.Column('float_col', FloatType()))
        t.update({'float_col': 1.0})
        float_col_vals = t.select(t.float_col).collect().to_pandas()['float_col']
        c3_vals = t.select(t.c3).collect().to_pandas()['c3']
        assert np.all(float_col_vals == pd.Series([1.0] * t.count()))
        t.update({'c3': t.float_col, 'float_col': t.c3})
        assert np.all(t.select(t.c3).collect().to_pandas()['c3'] == float_col_vals)
        assert np.all(t.select(t.float_col).collect().to_pandas()['float_col'] == c3_vals)
        t.revert()

        # update column that is used in computed cols
        t.add_column(catalog.Column('computed1', computed_with=t.c3 + 1))
        t.add_column(catalog.Column('computed2', computed_with=t.computed1 + 1))
        t.add_column(catalog.Column('computed3', computed_with=t.c3 + 3))

        # cascade=False
        computed1 = t.order_by(t.computed1).show(0).to_pandas()['computed1']
        computed2 = t.order_by(t.computed2).show(0).to_pandas()['computed2']
        computed3 = t.order_by(t.computed3).show(0).to_pandas()['computed3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1
        # update to a value that also satisfies the where clause
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=False)
        assert status.num_rows == 10
        assert status.updated_cols == ['test_tbl.c3']
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        # computed cols are not updated
        assert np.all(t.order_by(t.computed1).show(0).to_pandas()['computed1'] == computed1)
        assert np.all(t.order_by(t.computed2).show(0).to_pandas()['computed2'] == computed2)
        assert np.all(t.order_by(t.computed3).show(0).to_pandas()['computed3'] == computed3)

        # revert, then verify that we're back to where we started
        cl = pt.Client()
        t = cl.get_table(t.name)
        t.revert()
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 10.0).count() == 1

        # cascade=True
        status = t.update({'c3': 0.0}, where=t.c3 < 10.0, cascade=True)
        assert status.num_rows == 10
        assert set(status.updated_cols) == \
               set(['test_tbl.c3', 'test_tbl.computed1', 'test_tbl.computed2', 'test_tbl.computed3'])
        assert t.where(t.c3 < 10.0).count() == 10
        assert t.where(t.c3 == 0.0).count() == 10
        assert np.all(t.order_by(t.computed1).show(0).to_pandas()['computed1'][:10] == pd.Series([1.0] * 10))
        assert np.all(t.order_by(t.computed2).show(0).to_pandas()['computed2'][:10] == pd.Series([2.0] * 10))
        assert np.all(t.order_by(t.computed3).show(0).to_pandas()['computed3'][:10] == pd.Series([3.0] * 10))

        # bad update spec
        with pytest.raises(exc.Error) as excinfo:
            t.update({1: 1})
        assert 'dict key' in str(excinfo.value)

        # unknown column
        with pytest.raises(exc.Error) as excinfo:
            t.update({'unknown': 1})
        assert 'unknown unknown' in str(excinfo.value)

        # incompatible type
        with pytest.raises(exc.Error) as excinfo:
            t.update({'c1': 1})
        assert 'not compatible' in str(excinfo.value)

        # can't update primary key
        with pytest.raises(exc.Error) as excinfo:
            t.update({'c2': 1})
        assert 'primary key' in str(excinfo.value)

        # can't update computed column
        with pytest.raises(exc.Error) as excinfo:
            t.update({'computed1': 1})
        assert 'is computed' in str(excinfo.value)

        # non-expr
        with pytest.raises(exc.Error) as excinfo:
            t.update({'c3': lambda c3: math.sqrt(c3)})
        assert 'not a recognized' in str(excinfo.value)

        # non-Predicate filter
        with pytest.raises(exc.Error) as excinfo:
            t.update({'c3': 1.0}, where=lambda c2: c2 == 10)
        assert 'Predicate' in str(excinfo.value)

        img_t = indexed_img_tbl

        # can't update image col
        with pytest.raises(exc.Error) as excinfo:
            img_t.update({'img': 17}, where=img_t.img.nearest('car'))
        assert 'has type image' in str(excinfo.value)

        # similarity search is not supported
        with pytest.raises(exc.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.nearest('car'))
        assert 'nearest()' in str(excinfo.value)

        # filter not expressible in SQL
        with pytest.raises(exc.Error) as excinfo:
            img_t.update({'split': 'train'}, where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_cascading_update(self, test_tbl: pt.InsertableTable) -> None:
        t = test_tbl
        t.add_column(catalog.Column('d1', computed_with=t.c3 - 1))
        # add column that can be updated
        t.add_column(catalog.Column('c10', FloatType()))
        t.update({'c10': t.c3})
        # computed column that depends on two columns: exercise duplicate elimination during query construction
        t.add_column(catalog.Column('d2', computed_with=t.c3 - t.c10))
        r1 = t.where(t.c2 < 5).select(t.c3 + 1.0, t.c10 - 1.0, t.c3, 2.0).order_by(t.c2).show(0)
        t.update({'c4': True, 'c3': t.c3 + 1.0, 'c10': t.c10 - 1.0}, where=t.c2 < 5, cascade=True)
        r2 = t.where(t.c2 < 5).select(t.c3, t.c10, t.d1, t.d2).order_by(t.c2).show(0)
        assert_resultset_eq(r1, r2)

    def test_delete(self, test_tbl: pt.MutableTable, indexed_img_tbl: pt.MutableTable) -> None:
        t = test_tbl

        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 10
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1
        status = t.delete(where=t.c3 < 10.0)
        assert status.num_rows == 10
        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 0
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1

        # revert, then verify that we're back where we started
        cl = pt.Client()
        t = cl.get_table(t.name)
        t.revert()
        cnt = t.where(t.c3 < 10.0).count()
        assert cnt == 10
        cnt = t.where(t.c3 == 10.0).count()
        assert cnt == 1

        # non-Predicate filter
        with pytest.raises(exc.Error) as excinfo:
            t.delete(where=lambda c2: c2 == 10)
        assert 'Predicate' in str(excinfo.value)

        img_t = indexed_img_tbl
        # similarity search is not supported
        with pytest.raises(exc.Error) as excinfo:
            img_t.delete(where=img_t.img.nearest('car'))
        assert 'nearest()' in str(excinfo.value)

        # filter not expressible in SQL
        with pytest.raises(exc.Error) as excinfo:
            img_t.delete(where=img_t.img.width > 100)
        assert 'not expressible' in str(excinfo.value)

    def test_computed_cols(self, test_client: pt.client) -> None:
        cl = test_client
        c1 = catalog.Column('c1', IntType(nullable=False))
        c2 = catalog.Column('c2', FloatType(nullable=False))
        c3 = catalog.Column('c3', JsonType(nullable=False))
        schema = [c1, c2, c3]
        t = cl.create_table('test', schema)
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

        rows = create_table_data(t, ['c1', 'c2', 'c3'], num_rows=10)
        t.insert(rows, columns=['c1', 'c2', 'c3'])
        _ = t.show()

        # not allowed to pass values for computed cols
        with pytest.raises(exc.Error):
            rows2 = create_table_data(t, ['c1', 'c2', 'c3', 'c4'], num_rows=10)
            t.insert(rows2, columns=['c1', 'c2', 'c3', 'c4'])

        # computed col references non-existent col
        with pytest.raises(exc.Error):
            c1 = catalog.Column('c1', IntType(nullable=False))
            c2 = catalog.Column('c2', FloatType(nullable=False))
            c3 = catalog.Column('c3', FloatType(nullable=False), computed_with=lambda c2: math.sqrt(c2))
            _ = cl.create_table('test2', [c1, c3, c2])

        # test loading from store
        cl = pt.Client()
        t = cl.get_table('test')
        assert len(t.columns()) == len(t.columns())
        for i in range(len(t.columns())):
            if t.columns()[i].value_expr is not None:
                assert t.columns()[i].value_expr.equals(t.columns()[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t.insert(rows, columns=['c1', 'c2', 'c3'])
        res = t.show(0)
        tbl_df = t.show(0).to_pandas()

        # can't drop c4: c5 depends on it
        with pytest.raises(exc.Error):
            t.drop_column('c4')
        t.drop_column('c5')
        # now it works
        t.drop_column('c4')

    def test_computed_col_exceptions(self, test_client: pt.Client, test_tbl: catalog.MutableTable) -> None:
        cl = test_client

        # exception during insert()
        c2 = catalog.Column('c2', IntType(nullable=False))
        schema = [c2]
        rows = [list(r.values()) for r in list(test_tbl[test_tbl.c2].collect())]
        t = cl.create_table('test_insert', schema)
        _ = t.add_column(catalog.Column('add1', computed_with=self.f2(self.f1(t.c2))))
        status = t.insert(rows, columns=['c2'])
        assert status.num_excs == 10
        assert 'test_insert.add1' in status.cols_with_excs
        assert t[t.add1.errortype != None].count() == 10

        # exception during add_column()
        t = cl.create_table('test_add_column', schema)
        status = t.insert(rows, columns=['c2'])
        assert status.num_rows == 100
        assert status.num_excs == 0
        status = t.add_column(catalog.Column('add1', computed_with=self.f2(self.f1(t.c2))))
        assert status.num_excs == 10
        assert 'test_add_column.add1' in status.cols_with_excs
        assert t[t.add1.errortype != None].count() == 10

    def _test_computed_img_cols(self, t: catalog.MutableTable, stores_img_col: bool) -> None:
        rows, _ = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        status = t.insert([[r[0]] for r in rows[:20]], columns=['img'])
        assert status.num_rows == 20
        _ = t.count()
        _ = t.show()
        assert ImageStore.count(t.id) == t.count() * stores_img_col

        # test loading from store
        cl = pt.Client()
        t2 = cl.get_table(t.name)
        assert len(t.columns()) == len(t2.columns())
        for i in range(len(t.columns())):
            if t.columns()[i].value_expr is not None:
                assert t.columns()[i].value_expr.equals(t2.columns()[i].value_expr)

        # make sure we can still insert data and that computed cols are still set correctly
        t2.insert([[r[0]] for r in rows[:20]], columns=['img'])
        assert ImageStore.count(t2.id) == t2.count() * stores_img_col
        res = t2.show(0)
        tbl_df = t2.show(0).to_pandas()
        print(tbl_df)

        # revert also removes computed images
        t2.revert()
        assert ImageStore.count(t2.id) == t2.count() * stores_img_col

    def test_computed_img_cols(self, test_client: pt.Client) -> None:
        cl = test_client
        c1 = catalog.Column('img', ImageType(nullable=False), indexed=True)
        schema = [c1]
        t = cl.create_table('test', schema)
        t.add_column(catalog.Column('c2', computed_with=t.img.width))
        # c3 is not stored by default
        t.add_column(catalog.Column('c3', computed_with=t.img.rotate(90)))
        self._test_computed_img_cols(t, stores_img_col=False)

        t = cl.create_table('test2', schema)
        # c3 is now stored
        t.add_column(catalog.Column('c3', computed_with=t.img.rotate(90), stored=True))
        self._test_computed_img_cols(t, stores_img_col=True)
        _ = t[t.c3.errortype].show(0)

        # computed img col with exceptions
        t = cl.create_table('test3', schema)
        @pt.udf(return_type=ImageType(), param_types=[ImageType()])
        def f(img: PIL.Image.Image) -> PIL.Image.Image:
            raise RuntimeError
        t.add_column(catalog.Column('c3', computed_with=f(t.img), stored=True))
        rows, _ = read_data_file('imagenette2-160', 'manifest.csv', ['img'])
        t.insert([[r[0]] for r in rows[:20]], columns=['img'])
        _ = t[t.c3.errortype].show(0)

    def test_computed_window_fn(self, test_client: pt.Client, test_tbl: catalog.MutableTable) -> None:
        cl = test_client
        t = test_tbl
        # backfill
        t.add_column(catalog.Column('c9', computed_with=sum(t.c2, group_by=t.c4, order_by=t.c3)))

        c2 = catalog.Column('c2', IntType(nullable=False))
        c3 = catalog.Column('c3', FloatType(nullable=False))
        c4 = catalog.Column('c4', BoolType(nullable=False))
        new_t = cl.create_table('insert_test', [c2, c3, c4])
        new_t.add_column(catalog.Column('c5', IntType(), computed_with=lambda c2: c2 * c2))
        new_t.add_column(catalog.Column(
            'c6', computed_with=sum(new_t.c5, group_by=new_t.c4, order_by=new_t.c3)))
        rows = list(t.select(t.c2, t.c4, t.c3).collect())
        new_t.insert([list(r.values()) for r in rows], columns=['c2', 'c4', 'c3'])
        _ = new_t.show(0)
        print(_)

    def test_revert(self, test_client: pt.Client) -> None:
        cl = test_client
        t1 = make_tbl(cl, 'test1', ['c1', 'c2'])
        assert t1.version() == 0
        rows1 = create_table_data(t1)
        t1.insert(rows1, columns=['c1', 'c2'])
        assert t1.count() == len(rows1)
        assert t1.version() == 1
        rows2 = create_table_data(t1)
        t1.insert(rows2, columns=['c1', 'c2'])
        assert t1.count() == len(rows1) + len(rows2)
        assert t1.version() == 2
        t1.revert()
        assert t1.count() == len(rows1)
        assert t1.version() == 1
        t1.insert(rows2, columns=['c1', 'c2'])
        assert t1.count() == len(rows1) + len(rows2)
        assert t1.version() == 2

        # can't revert past version 0
        t1.revert()
        t1.revert()
        with pytest.raises(exc.Error) as excinfo:
            t1.revert()
        assert 'version 0' in str(excinfo.value)

    def test_snapshot(self, test_client: pt.Client) -> None:
        cl = test_client
        cl.create_dir('main')
        tbl = make_tbl(cl, 'main.test1', ['c1', 'c2'])
        rows1 = create_table_data(tbl)
        tbl.insert(rows1, columns=['c1', 'c2'])
        assert tbl.count() == len(rows1)

        cl.create_dir('snap')
        cl.create_snapshot('snap.test1', 'main.test1')
        snap = cl.get_table('snap.test1')
        assert cl.get_path(snap) == 'snap.test1'
        assert snap.count() == len(rows1)

        # reload md
        cl = pt.Client()
        snap = cl.get_table('snap.test1')
        assert snap.count() == len(rows1)

        # adding data to a base table doesn't change the snapshot
        rows2 = create_table_data(tbl)
        tbl.insert(rows2, columns=['c1', 'c2'])
        assert tbl.count() == len(rows1) + len(rows2)
        assert snap.count() == len(rows1)

        tbl.revert()
        # can't revert a version referenced by a snapshot
        with pytest.raises(exc.Error) as excinfo:
            tbl.revert()
        assert 'version is needed' in str(excinfo.value)

        # can't drop a table with snapshots
        with pytest.raises(exc.Error) as excinfo:
            cl.drop_table('main.test1')
        assert 'snapshot' in str(excinfo.value)

    def test_add_column(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.add_column(catalog.Column('add1', pt.IntType(nullable=False)))
        assert len(t.columns()) == num_orig_cols + 1

        # duplicate name
        with pytest.raises(exc.Error) as exc_info:
            t.add_column(catalog.Column('c1', pt.IntType()))
        assert 'duplicate column name' in str(exc_info.value).lower()
        # bad name
        with pytest.raises(exc.Error) as exc_info:
            t.add_column(catalog.Column('bad name', pt.IntType()))
        assert 'invalid column name' in str(exc_info.value).lower()

        # make sure this is still true after reloading the metadata
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert len(t.columns()) == num_orig_cols + 1

        # revert() works
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert len(t.columns()) == num_orig_cols

    def test_drop_column(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.drop_column('c1')
        assert len(t.columns()) == num_orig_cols - 1

        with pytest.raises(exc.Error):
            t.drop_column('unknown')

        # make sure this is still true after reloading the metadata
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert len(t.columns()) == num_orig_cols - 1

        # revert() works
        t.revert()
        assert len(t.columns()) == num_orig_cols

        # make sure this is still true after reloading the metadata once more
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert len(t.columns()) == num_orig_cols

    def test_rename_column(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl
        num_orig_cols = len(t.columns())
        t.rename_column('c1', 'c1_renamed')
        assert len(t.columns()) == num_orig_cols

        # unknown column
        with pytest.raises(exc.Error):
            t.rename_column('unknown', 'unknown_renamed')
        # bad name
        with pytest.raises(exc.Error):
            t.rename_column('c2', 'bad name')
        # existing name
        with pytest.raises(exc.Error):
            t.rename_column('c2', 'c3')

        # make sure this is still true after reloading the metadata
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert 'c1' not in t.cols_by_name
        assert 'c1_renamed' in t.cols_by_name

        # revert() works
        t.revert()
        assert 'c1' in t.cols_by_name
        assert 'c1_renamed' not in t.cols_by_name

        # make sure this is still true after reloading the metadata once more
        cl = pt.Client()
        t = cl.get_table(t.name)
        assert 'c1' in t.cols_by_name
        assert 'c1_renamed' not in t.cols_by_name

    def test_add_computed_column(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl
        status = t.add_column(catalog.Column('add1', computed_with=t.c2 + 10))
        assert status.num_excs == 0
        _ = t.show()

        # with exception in SQL
        with pytest.raises(exc.Error):
            t.add_column(catalog.Column('add2', computed_with=(t.c2 - 10) / (t.c3 - 10)))

        # with exception in Python for c6.f2 == 10
        status = t.add_column(catalog.Column('add2', computed_with=(t.c6.f2 - 10) / (t.c6.f2 - 10)))
        assert status.num_excs == 1
        result = t[t.add2.errortype != None][t.c6.f2, t.add2, t.add2.errortype, t.add2.errormsg].show()
        assert len(result) == 1

        # test case: exceptions in dependencies prevent execution of dependent exprs
        status = t.add_column(catalog.Column('add3', computed_with=self.f2(self.f1(t.c2))))
        assert status.num_excs == 10
        result = t[t.add3.errortype != None][t.c2, t.add3, t.add3.errortype, t.add3.errormsg].show()
        assert len(result) == 10

    def test_describe(self, test_tbl: catalog.MutableTable) -> None:
        t = test_tbl
        fn = lambda c2: np.full((3, 4), c2)
        t.add_column(
            catalog.Column('computed1', col_type=ArrayType((3, 4), dtype=IntType()), computed_with=fn))
        t.describe()

        # TODO: how to you check the output of these?
        _ = t.__repr__()
        _ = t._repr_html_()

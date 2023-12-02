import pytest
import math
import numpy as np
import pandas as pd
import datetime

import PIL

import pixeltable as pt
from pixeltable import exceptions as exc
from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import create_test_tbl, assert_resultset_eq, get_video_files
from pixeltable.iterators import FrameIterator


class TestComponentView:
    def test_basic(self, test_client: pt.Client) -> None:
        cl = test_client
        # create video table
        cols = [catalog.Column('video', VideoType()), catalog.Column('angle', IntType())]
        video_t = cl.create_table('video_tbl', cols)
        video_filepaths = get_video_files()

        # cannot add 'pos' column
        with pytest.raises(exc.Error) as excinfo:
            video_t.add_column(catalog.Column('pos', IntType()))
        assert 'reserved' in str(excinfo.value)

        # parameter missing
        with pytest.raises(exc.Error) as excinfo:
            args = {'fps': 1}
            _ = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        assert 'missing a required argument' in str(excinfo.value)

        # bad parameter type
        with pytest.raises(exc.Error) as excinfo:
            args = {'video': video_t.video, 'fps': '1'}
            _ = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        assert 'expected int' in str(excinfo.value)

        # bad parameter type
        with pytest.raises(exc.Error) as excinfo:
            args = {'video': 1, 'fps': 1}
            _ = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        assert 'expected file path' in str(excinfo.value)

        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        view_t = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        # computed column that references an unstored computed column from the view and a column from the base
        view_t.add_column(catalog.Column('angle2', computed_with=view_t.angle + 1))
        # computed column that references an unstored and a stored computed view column
        view_t.add_column(catalog.Column('v1', computed_with=view_t.frame.rotate(view_t.angle2), stored=True))
        # computed column that references a stored computed column from the view
        view_t.add_column(catalog.Column('v2', computed_with=view_t.frame_idx - 1))

        # and load data
        rows = [[p, 30] for p in video_filepaths]
        video_t.insert(rows)
        # pos and frame_idx are identical
        res = view_t.select(view_t.pos, view_t.frame_idx).collect().to_pandas()
        assert np.all(res['pos'] == res['frame_idx'])

        video_url = video_t.select(video_t.video.fileurl).show(0)[0, 0]
        result = view_t.where(view_t.video == video_url).select(view_t.frame, view_t.frame_idx) \
            .collect()
        result = view_t.where(view_t.video == video_url).select(view_t.frame_idx).order_by(view_t.frame_idx) \
            .collect().to_pandas()
        assert len(result) > 0
        assert np.all(result['frame_idx'] == pd.Series(range(len(result))))

    def test_add_column(self, test_client: pt.Client) -> None:
        cl = test_client
        # create video table
        video_t = cl.create_table('video_tbl', [catalog.Column('video', VideoType())])
        video_filepaths = get_video_files()
        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        view_t = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)

        rows = [[p] for p in video_filepaths]
        video_t.insert(rows)
        # adding a non-computed column backfills it with nulls
        view_t.add_column(catalog.Column('annotation', JsonType(nullable=True)))
        assert view_t.count() == view_t.where(view_t.annotation == None).count()
        # adding more data via the base table sets the column values to null
        video_t.insert(rows)
        _ = view_t.where(view_t.annotation == None).count()
        assert view_t.count() == view_t.where(view_t.annotation == None).count()

        with pytest.raises(exc.Error) as excinfo:
            view_t.add_column(catalog.Column('annotation', JsonType(nullable=False)))
        assert 'must be nullable' in str(excinfo.value)

    def test_update(self, test_client: pt.Client) -> None:
        cl = test_client
        # create video table
        video_t = cl.create_table('video_tbl', [catalog.Column('video', VideoType())])
        # create frame view with manually updated column
        args = {'video': video_t.video, 'fps': 1}
        dict_col = catalog.Column('annotation', JsonType(nullable=True))
        view_t = cl.create_view(
            'test_view', video_t, schema=[dict_col], iterator_class=FrameIterator, iterator_args=args)

        video_filepaths = get_video_files()
        rows = [[p] for p in video_filepaths]
        video_t.insert(rows)
        import urllib
        video_url = urllib.parse.urljoin('file:', urllib.request.pathname2url(video_filepaths[0]))
        view_t.update({'annotation': {'a': 1}}, where=view_t.video == video_url)
        c1 = view_t.where(view_t.annotation != None).count()
        c2 = view_t.where(view_t.video == video_url).count()
        assert c1 == c2

        with pytest.raises(exc.Error) as excinfo:
            non_nullable_col = catalog.Column('annotation', JsonType(nullable=False))
            _ = cl.create_view(
                'bad_view', video_t, schema=[non_nullable_col], iterator_class=FrameIterator, iterator_args=args)
        assert 'must be nullable' in str(excinfo.value)

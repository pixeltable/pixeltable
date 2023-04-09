from typing import Optional
import pytest
import PIL

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import exceptions as exc
from pixeltable.utils.filecache import FileCache
from pixeltable.utils.imgstore import ImageStore
from pixeltable.utils.video import num_tmp_frames


class TestVideo:
    def test_basic(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db

        def create_and_insert(stored: Optional[bool]) -> catalog.Table:
            FileCache.get().clear()
            cols = [
                catalog.Column('video', VideoType(), nullable=False),
                catalog.Column('frame', ImageType(), nullable=False, stored=stored, indexed=True),
                catalog.Column('frame_idx', IntType(), nullable=False),
            ]
            # extract frames at fps=1
            db.drop_table('test', ignore_errors=True)
            tbl = db.create_table(
                'test', cols, extract_frames_from='video', extracted_frame_col='frame',
                extracted_frame_idx_col='frame_idx', extracted_fps=1)
            assert num_tmp_frames() == 0
            tbl.insert_rows([[p] for p in video_filepaths[:2]], columns=['video'])
            assert num_tmp_frames() == 0
            total_num_rows = tbl.count()
            result = tbl[tbl.frame_idx >= 5][tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(0)
            assert len(result) == total_num_rows - 2 * 5
            result = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(3)
            assert len(result) == 3
            result = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(0)
            assert len(result) == total_num_rows
            assert num_tmp_frames() == 0
            return tbl

        # default case: extracted frames are cached but not stored
        tbl = create_and_insert(None)
        assert ImageStore.count(tbl.id) == 0
        assert FileCache.get().num_files() == tbl.count()

        # extracted frames are neither stored nor cached
        tbl = create_and_insert(False)
        assert ImageStore.count(tbl.id) == 0
        assert FileCache.get().num_files() == 0

        # extracted frames are stored
        tbl = create_and_insert(True)
        assert ImageStore.count(tbl.id) == tbl.count()
        assert FileCache.get().num_files() == 0
        # revert() also removes extracted frames
        tbl.insert_rows([[p] for p in video_filepaths], columns=['video'])
        tbl.revert()
        assert ImageStore.count(tbl.id) == tbl.count()

        # missing 'columns' arg
        with pytest.raises(exc.Error):
            tbl.insert_rows([[p] for p in video_filepaths])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2], [3]], columns=['video'])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2]], columns=['video'])

        # create snapshot to make sure we can still retrieve frames
        db.create_snapshot('snap', ['test'])
        snap = db.get_table('snap.test')
        _ = snap[snap.frame].show(10)

        # fps=1 expressed as a filter
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        tbl2 = db.create_table(
            'test2', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=0,
            ffmpeg_filter={'select': 'isnan(prev_selected_t)+gte(t-prev_selected_t, 1)'})
        tbl2.insert_rows([[p] for p in video_filepaths[:2]], columns=['video'])
        # for some reason there's one extra frame in tbl2
        assert tbl.count() == tbl2.count() - 1

    def test_cached_cols(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        # all image cols are stored=None by default
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        t = db.create_table(
            'test', cols, extract_frames_from = 'video', extracted_frame_col = 'frame',
            extracted_frame_idx_col = 'frame_idx', extracted_fps = 1)
        # c2 and c4 depend directly on c1, c3 depends on it indirectly
        t.add_column(catalog.Column('c1', computed_with=t.frame.resize((224, 224))))
        t.add_column(catalog.Column('c2', computed_with=t.c1.rotate(10)))
        t.add_column(catalog.Column('c3', computed_with=t.c2.rotate(20)))
        t.add_column(catalog.Column('c4', computed_with=t.c1.rotate(30)))
        cache_stats = FileCache.get().stats()
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert not t.cols_by_name[name].is_stored
        t.insert_rows([[p] for p in video_filepaths], columns=['video'])
        cache_stats = FileCache.get().stats()
        _ = t[t.c1, t.c2, t.c3, t.c4].show(0)

        # the query populated the cache
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == t.count() * 5  # 5 cached cols
        assert cache_stats.num_hits == 0  # nothing cached on first access

        # at this point, all requests should be served from the cache, and we're accessing 4 cached cols
        # (not 5: the frame col doesn't need to be accessed)
        _ = t[t.c1, t.c2, t.c3, t.c4].show(0)
        cache_stats = FileCache.get().stats()
        assert cache_stats.num_requests == t.count() * (5 + 4)
        assert cache_stats.num_hits == t.count() * 4

        cl = pt.Client()
        _ = cl.get_db('test')
        _ = cl.cache_stats()
        _ = cl.cache_util()
        print(_)

    def test_make_video(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        t = db.create_table(
            'test', cols, extract_frames_from = 'video', extracted_frame_col = 'frame',
            extracted_frame_idx_col = 'frame_idx', extracted_fps = 1)
        t.insert_rows([[p] for p in video_filepaths], columns=['video'])
        _ = t[pt.make_video(t.frame_idx, t.frame)].group_by(t.video).show()
        print(_)

        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = t[pt.make_video(t.frame_idx, t.frame, group_by=t.video)].show()
        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = t[pt.make_video(t.frame, order_by=t.frame_idx)].show()

        class WindowAgg:
            def __init__(self):
                self.sum = 0
            @classmethod
            def make_aggregator(cls) -> 'WindowAgg':
                return cls()
            def update(self, frame: PIL.Image.Image) -> None:
                self.sum += 1
            def value(self) -> int:
                return self.sum

        agg_fn = pt.make_aggregate_function(
            IntType(), [ImageType()],
            init_fn=WindowAgg.make_aggregator,
            update_fn=WindowAgg.update,
            value_fn=WindowAgg.value,
            requires_order_by=True, allows_std_agg=False, allows_window=True)
        # make sure it works
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        db.create_function('agg_fn', agg_fn)

        # reload from store
        cl = pt.Client()
        db = cl.get_db('test')
        agg_fn = db.get_function('agg_fn')
        t = db.get_table('test')
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        print(_)

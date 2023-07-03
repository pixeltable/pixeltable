from typing import Optional, List
import pytest
import PIL

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import exceptions as exc
from pixeltable.utils.imgstore import ImageStore


class TestVideo:
    def create_and_insert(self, db: catalog.Db, stored: Optional[bool], paths: List[str]) -> catalog.Table:
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        # extract frames at fps=1
        db.drop_table('test', ignore_errors=True)
        tbl = db.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=1)
        tbl.add_column(catalog.Column('transform', computed_with=tbl.frame.rotate(90), stored=stored))
        tbl.insert_rows([[p] for p in paths], columns=['video'])
        total_num_rows = tbl.count()
        result = tbl[tbl.frame_idx >= 5][tbl.frame_idx, tbl.frame, tbl.transform].show(0)
        assert len(result) == total_num_rows - len(paths) * 5
        result = tbl[tbl.frame_idx, tbl.frame, tbl.transform].show(3)
        assert len(result) == 3
        result = tbl[tbl.frame_idx, tbl.frame, tbl.transform].show(0)
        assert len(result) == total_num_rows
        return tbl

    def test_basic(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db

        # default case: extracted frames are not stored
        tbl = self.create_and_insert(db, None, video_filepaths)
        assert ImageStore.count(tbl.id) == 0

        # extracted frames are explicitly not stored
        tbl = self.create_and_insert(db, False, video_filepaths)
        assert ImageStore.count(tbl.id) == 0

        # extracted frames are stored
        tbl = self.create_and_insert(db, True, video_filepaths)
        assert ImageStore.count(tbl.id) == tbl.count()
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
        db.create_snapshot('snap', 'test')
        snap = db.get_table('snap')
        _ = snap[snap.frame].show(10)

    def test_computed_cols(self, test_db: catalog.Db) -> None:
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
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert not t.cols_by_name[name].is_stored
        t.insert_rows([[p] for p in video_filepaths], columns=['video'])
        _ = t[t.c1, t.c2, t.c3, t.c4].show(0)

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
                self.img = None
            @classmethod
            def make_aggregator(cls) -> 'WindowAgg':
                return cls()
            def update(self, frame: PIL.Image.Image) -> None:
                self.img = frame
            def value(self) -> PIL.Image.Image:
                return self.img

        agg_fn = pt.make_aggregate_function(
            ImageType(), [ImageType()],
            init_fn=WindowAgg.make_aggregator,
            update_fn=WindowAgg.update,
            value_fn=WindowAgg.value,
            requires_order_by=True, allows_std_agg=False, allows_window=True)
        # make sure it works
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        db.create_function('agg_fn', agg_fn)
        t.add_column(catalog.Column('agg', computed_with=agg_fn(t.frame_idx, t.frame, group_by=t.video)))
        assert t.cols_by_name['agg'].is_stored
        _ = t[pt.make_video(t.frame_idx, t.agg)].group_by(t.video).show()
        print(_)

        # image cols computed with a window function currently need to be stored
        with pytest.raises(exc.Error):
            t.add_column(
                catalog.Column('agg2', computed_with=agg_fn(t.frame_idx, t.frame, group_by=t.video), stored=False))

        # reload from store
        cl = pt.Client()
        db = cl.get_db('test')
        agg_fn = db.get_function('agg_fn')
        t = db.get_table('test')
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        print(_)

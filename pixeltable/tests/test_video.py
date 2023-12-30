from typing import Optional, List, Tuple
import pytest
import PIL

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import exceptions as exc
from pixeltable.utils.media_store import MediaStore
from pixeltable.iterators import FrameIterator


class TestVideo:
    def create_tbls(
            self, cl: pt.Client, base_name: str = 'video_tbl', view_name: str = 'frame_view'
    ) -> Tuple[catalog.InsertableTable, catalog.MutableTable]:
        cl.drop_table(view_name, ignore_errors=True)
        cl.drop_table(base_name, ignore_errors=True)
        base_t = cl.create_table(base_name, [catalog.Column('video', VideoType())])
        args = {'video': base_t.video, 'fps': 1}
        view_t = cl.create_view(view_name, base_t, iterator_class=FrameIterator, iterator_args=args)
        return base_t, view_t

    def create_and_insert(
            self, cl: pt.Client, stored: Optional[bool], paths: List[str]
    ) -> Tuple[catalog.InsertableTable, catalog.MutableTable]:
        base_t, view_t = self.create_tbls(cl)

        view_t.add_column(catalog.Column('transform', computed_with=view_t.frame.rotate(90), stored=stored))
        base_t.insert([{'video': p} for p in paths])
        total_num_rows = view_t.count()
        result = view_t[view_t.frame_idx >= 5][view_t.frame_idx, view_t.frame, view_t.transform].show(0)
        assert len(result) == total_num_rows - len(paths) * 5
        result = view_t[view_t.frame_idx, view_t.frame, view_t.transform].show(3)
        assert len(result) == 3
        result = view_t[view_t.frame_idx, view_t.frame, view_t.transform].show(0)
        assert len(result) == total_num_rows
        return base_t, view_t

    def test_basic(self, test_client: pt.Client) -> None:
        video_filepaths = get_video_files()
        cl = test_client

        # default case: computed images are not stored
        _, view = self.create_and_insert(cl, None, video_filepaths)
        assert MediaStore.count(view.id) == 0

        # computed images are explicitly not stored
        _, view = self.create_and_insert(cl, False, video_filepaths)
        assert MediaStore.count(view.id) == 0

        # computed images are stored
        tbl, view = self.create_and_insert(cl, True, video_filepaths)
        assert MediaStore.count(view.id) == view.count()

        # revert() also removes computed images
        tbl.insert([{'video': p} for p in video_filepaths])
        tbl.revert()
        assert MediaStore.count(view.id) == view.count()

    def test_query(self, test_client: pt.client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        base_t.insert([{'video': p} for p in video_filepaths])
        res = view_t.where(view_t.video == video_filepaths[0]).show(0)

    def test_computed_cols(self, test_client: pt.client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        # c2 and c4 depend directly on c1, c3 depends on it indirectly
        view_t.add_column(catalog.Column('c1', computed_with=view_t.frame.resize([224, 224])))
        view_t.add_column(catalog.Column('c2', computed_with=view_t.c1.rotate(10)))
        view_t.add_column(catalog.Column('c3', computed_with=view_t.c2.rotate(20)))
        view_t.add_column(catalog.Column('c4', computed_with=view_t.c1.rotate(30)))
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert not view_t.cols_by_name[name].is_stored
        base_t.insert([{'video': p} for p in video_filepaths])
        _ = view_t[view_t.c1, view_t.c2, view_t.c3, view_t.c4].show(0)

    def test_make_video(self, test_client: pt.Client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        base_t.insert([{'video': p} for p in video_filepaths])
        # reference to the frame col requires ordering by base, pos
        _ = view_t.select(pt.make_video(view_t.pos, view_t.frame)).group_by(base_t).show()
        # the same without frame col
        view_t.add_column(catalog.Column('transformed', computed_with=view_t.frame.rotate(30), stored=True))
        _ = view_t.select(pt.make_video(view_t.pos, view_t.transformed)).group_by(base_t).show()

        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = view_t.select(pt.make_video(view_t.pos, view_t.frame, group_by=base_t)).show()
        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = view_t.select(pt.make_video(view_t.frame, order_by=view_t.pos)).show()
        with pytest.raises(exc.Error):
            # incompatible ordering requirements
            _ = view_t.select(
                pt.make_video(view_t.pos, view_t.frame),
                pt.make_video(view_t.pos - 1, view_t.transformed)).group_by(base_t).show()

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
        _ = view_t.select(agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()
        cl.create_function('agg_fn', agg_fn)
        view_t.add_column(
            catalog.Column('agg', computed_with=agg_fn(view_t.pos, view_t.frame, group_by=base_t)))
        assert view_t.cols_by_name['agg'].is_stored
        _ = view_t.select(pt.make_video(view_t.pos, view_t.agg)).group_by(base_t).show()

        # image cols computed with a window function currently need to be stored
        with pytest.raises(exc.Error):
            view_t.add_column(
                catalog.Column('agg2', computed_with=agg_fn(view_t.pos, view_t.frame, group_by=base_t), stored=False))

        # reload from store
        cl = pt.Client()
        agg_fn = cl.get_function('agg_fn')
        base_t, view_t = cl.get_table(base_t.name), cl.get_table(view_t.name)
        _ = view_t.select(agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()
        print(_)

from typing import Optional, List, Tuple

import PIL
import pytest

import pixeltable as pxt
from pixeltable import catalog
from pixeltable import exceptions as excs
from pixeltable.iterators import FrameIterator
from pixeltable.tests.utils import get_video_files
from pixeltable.tests.utils import skip_test_if_not_installed
from pixeltable.type_system import VideoType, ImageType
from pixeltable.utils.media_store import MediaStore


class TestVideo:
    def create_tbls(
            self, cl: pxt.Client, base_name: str = 'video_tbl', view_name: str = 'frame_view'
    ) -> Tuple[catalog.InsertableTable, catalog.Table]:
        cl.drop_table(view_name, ignore_errors=True)
        cl.drop_table(base_name, ignore_errors=True)
        base_t = cl.create_table(base_name, {'video': VideoType()})
        args = {'video': base_t.video, 'fps': 1}
        view_t = cl.create_view(view_name, base_t, iterator_class=FrameIterator, iterator_args=args)
        return base_t, view_t

    def create_and_insert(
            self, cl: pxt.Client, stored: Optional[bool], paths: List[str]
    ) -> Tuple[catalog.InsertableTable, catalog.Table]:
        base_t, view_t = self.create_tbls(cl)

        view_t.add_column(transform=view_t.frame.rotate(90), stored=stored)
        base_t.insert({'video': p} for p in paths)
        total_num_rows = view_t.count()
        result = view_t[view_t.frame_idx >= 5][view_t.frame_idx, view_t.frame, view_t.transform].show(0)
        assert len(result) == total_num_rows - len(paths) * 5
        result = view_t[view_t.frame_idx, view_t.frame, view_t.transform].show(3)
        assert len(result) == 3
        result = view_t[view_t.frame_idx, view_t.frame, view_t.transform].show(0)
        assert len(result) == total_num_rows
        return base_t, view_t

    def test_basic(self, test_client: pxt.Client) -> None:
        video_filepaths = get_video_files()
        cl = test_client

        # default case: computed images are not stored
        _, view = self.create_and_insert(cl, None, video_filepaths)
        assert MediaStore.count(view.get_id()) == 0

        # computed images are explicitly not stored
        _, view = self.create_and_insert(cl, False, video_filepaths)
        assert MediaStore.count(view.get_id()) == 0

        # computed images are stored
        tbl, view = self.create_and_insert(cl, True, video_filepaths)
        assert MediaStore.count(view.get_id()) == view.count()

        # revert() also removes computed images
        tbl.insert({'video': p} for p in video_filepaths)
        tbl.revert()
        assert MediaStore.count(view.get_id()) == view.count()

    def test_query(self, test_client: pxt.client) -> None:
        skip_test_if_not_installed('boto3')
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        # also include an external file, to make sure that prefetching works
        url = 's3://multimedia-commons/data/videos/mp4/ffe/ff3/ffeff3c6bf57504e7a6cecaff6aefbc9.mp4'
        video_filepaths.append(url)
        status = base_t.insert({'video': p} for p in video_filepaths)
        assert status.num_excs == 0
        # make sure that we can get the frames back
        res = view_t.select(view_t.frame).collect().to_pandas()
        assert res['frame'].notnull().all()
        # make sure we can select a specific video
        all_rows = view_t.select(url=view_t.video.fileurl).collect().to_pandas()
        res = view_t.where(view_t.video == url).collect()
        assert len(res) == len(all_rows[all_rows.url == url])

    def test_fps(self, test_client: pxt.client) -> None:
        cl = test_client
        path = get_video_files()[0]
        videos = cl.create_table('videos', {'video': VideoType()})
        frames_1_0 = cl.create_view(
            'frames_1_0', videos, iterator_class=FrameIterator, iterator_args={'video': videos.video, 'fps': 1})
        frames_0_5 = cl.create_view(
            'frames_0_5', videos, iterator_class=FrameIterator, iterator_args={'video': videos.video, 'fps': 1/2})
        frames_0_33 = cl.create_view(
            'frames_0_33', videos, iterator_class=FrameIterator, iterator_args={'video': videos.video, 'fps': 1/3})
        videos.insert(video=path)
        assert frames_0_5.count() == frames_1_0.count() // 2 or frames_0_5.count() == frames_1_0.count() // 2 + 1
        assert frames_0_33.count() == frames_1_0.count() // 3 or frames_0_33.count() == frames_1_0.count() // 3 + 1

    def test_computed_cols(self, test_client: pxt.client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        # c2 and c4 depend directly on c1, c3 depends on it indirectly
        view_t.add_column(c1=view_t.frame.resize([224, 224]))
        view_t.add_column(c2=view_t.c1.rotate(10))
        view_t.add_column(c3=view_t.c2.rotate(20))
        view_t.add_column(c4=view_t.c1.rotate(30))
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert not view_t.tbl_version_path.tbl_version.cols_by_name[name].is_stored
        base_t.insert({'video': p} for p in video_filepaths)
        _ = view_t[view_t.c1, view_t.c2, view_t.c3, view_t.c4].show(0)

    # window function that simply passes through the frame
    @pxt.uda(
        update_types=[ImageType()], value_type=ImageType(),
        requires_order_by=True, allows_std_agg=False, allows_window=True)
    class agg_fn:
        def __init__(self):
            self.img = None
        def update(self, frame: PIL.Image.Image) -> None:
            self.img = frame
        def value(self) -> PIL.Image.Image:
            return self.img

    def test_make_video(self, test_client: pxt.Client) -> None:
        video_filepaths = get_video_files()
        cl = test_client
        base_t, view_t = self.create_tbls(cl)
        base_t.insert({'video': p} for p in video_filepaths)
        # reference to the frame col requires ordering by base, pos
        from pixeltable.functions import make_video
        _ = view_t.select(make_video(view_t.pos, view_t.frame)).group_by(base_t).show()
        # the same without frame col
        view_t.add_column(transformed=view_t.frame.rotate(30), stored=True)
        _ = view_t.select(make_video(view_t.pos, view_t.transformed)).group_by(base_t).show()

        with pytest.raises(excs.Error):
            # make_video() doesn't allow windows
            _ = view_t.select(make_video(view_t.pos, view_t.frame, group_by=base_t)).show()
        with pytest.raises(excs.Error):
            # make_video() requires ordering
            _ = view_t.select(make_video(view_t.frame, order_by=view_t.pos)).show()
        with pytest.raises(excs.Error):
            # incompatible ordering requirements
            _ = view_t.select(
                make_video(view_t.pos, view_t.frame),
                make_video(view_t.pos - 1, view_t.transformed)).group_by(base_t).show()

        # make sure it works
        _ = view_t.select(self.agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()
        status = view_t.add_column(agg=self.agg_fn(view_t.pos, view_t.frame, group_by=base_t))
        assert status.num_excs == 0
        _ = view_t.select(make_video(view_t.pos, view_t.agg)).group_by(base_t).show()

        # image cols computed with a window function currently need to be stored
        with pytest.raises(excs.Error):
            view_t.add_column(agg2=self.agg_fn(view_t.pos, view_t.frame, group_by=base_t), stored=False)

        # reload from store
        cl = pxt.Client(reload=True)
        base_t, view_t = cl.get_table(base_t.get_name()), cl.get_table(view_t.get_name())
        _ = view_t.select(self.agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()

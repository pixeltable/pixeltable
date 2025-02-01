from typing import Optional

import PIL
import pytest

import pixeltable as pxt
from pixeltable import exceptions as excs
from pixeltable.iterators import FrameIterator
from pixeltable.utils.media_store import MediaStore

from .utils import get_video_files, reload_catalog, skip_test_if_not_installed, validate_update_status


class TestVideo:
    def create_tbls(self, base_name: str = 'video_tbl', view_name: str = 'frame_view') -> tuple[pxt.Table, pxt.Table]:
        pxt.drop_table(view_name, if_not_exists='ignore')
        pxt.drop_table(base_name, if_not_exists='ignore')
        base_t = pxt.create_table(base_name, {'video': pxt.Video})
        view_t = pxt.create_view(view_name, base_t, iterator=FrameIterator.create(video=base_t.video, fps=1))
        return base_t, view_t

    def create_and_insert(self, stored: Optional[bool], paths: list[str]) -> tuple[pxt.Table, pxt.Table]:
        base_t, view_t = self.create_tbls()

        view_t.add_computed_column(transform=view_t.frame.rotate(90), stored=stored)
        base_t.insert({'video': p} for p in paths)
        total_num_rows = view_t.count()
        result = view_t.where(view_t.frame_idx >= 5).select(view_t.frame_idx, view_t.frame, view_t.transform).collect()
        assert len(result) == total_num_rows - len(paths) * 5
        result = view_t.select(view_t.frame_idx, view_t.frame, view_t.transform).show(3)
        assert len(result) == 3
        result = view_t.select(view_t.frame_idx, view_t.frame, view_t.transform).collect()
        assert len(result) == total_num_rows
        # Try inserting a row with a `None` video; confirm that it produces no additional rows in the view
        base_t.insert(video=None)
        result = view_t.select(view_t.frame_idx, view_t.frame, view_t.transform).collect()
        assert len(result) == total_num_rows
        return base_t, view_t

    def test_basic(self, reset_db) -> None:
        video_filepaths = get_video_files()

        # computed images are not stored
        _, view = self.create_and_insert(False, video_filepaths)
        assert MediaStore.count(view._id) == 0

        # computed images are stored
        tbl, view = self.create_and_insert(True, video_filepaths)
        assert MediaStore.count(view._id) == view.count()

        # revert() also removes computed images
        tbl.insert({'video': p} for p in video_filepaths)
        tbl.revert()
        assert MediaStore.count(view._id) == view.count()

    def test_query(self, reset_db) -> None:
        skip_test_if_not_installed('boto3')
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls()
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

    def test_fps(self, reset_db) -> None:
        path = get_video_files()[0]
        videos = pxt.create_table('videos', {'video': pxt.Video})
        frames_all = pxt.create_view('frames_all', videos, iterator=FrameIterator.create(video=videos.video))
        frames_1_0 = pxt.create_view('frames_1_0', videos, iterator=FrameIterator.create(video=videos.video, fps=1))
        frames_0_5 = pxt.create_view('frames_0_5', videos, iterator=FrameIterator.create(video=videos.video, fps=1 / 2))
        frames_0_33 = pxt.create_view(
            'frames_0_33', videos, iterator=FrameIterator.create(video=videos.video, fps=1 / 3)
        )
        num_frames_10 = pxt.create_view(
            'num_frames_10', videos, iterator=FrameIterator.create(video=videos.video, num_frames=10)
        )
        num_frames_50 = pxt.create_view(
            'num_frames_50', videos, iterator=FrameIterator.create(video=videos.video, num_frames=50)
        )
        num_frames_1000 = pxt.create_view(
            'num_frames_1000', videos, iterator=FrameIterator.create(video=videos.video, num_frames=1000)
        )
        videos.insert(video=path)
        assert frames_all.count() == 449
        assert frames_1_0.count() == 15
        assert frames_0_5.count() == 8
        assert frames_0_33.count() == 5
        assert num_frames_10.count() == 10
        assert num_frames_50.count() == 50
        assert num_frames_1000.count() == 449
        with pytest.raises(excs.Error) as exc_info:
            _ = pxt.create_view(
                'invalid_args', videos, iterator=FrameIterator.create(video=videos.video, fps=1 / 2, num_frames=10)
            )
        assert 'At most one of `fps` or `num_frames` may be specified' in str(exc_info.value)

    def test_computed_cols(self, reset_db) -> None:
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls()
        # c2 and c4 depend directly on c1, c3 depends on it indirectly
        view_t.add_computed_column(c1=view_t.frame.resize([224, 224]))
        view_t.add_computed_column(c2=view_t.c1.rotate(10))
        view_t.add_computed_column(c3=view_t.c2.rotate(20))
        view_t.add_computed_column(c4=view_t.c1.rotate(30))
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert view_t._tbl_version_path.tbl_version.cols_by_name[name].is_stored
        base_t.insert({'video': p} for p in video_filepaths)
        _ = view_t.select(view_t.c1, view_t.c2, view_t.c3, view_t.c4).collect()

    def test_get_metadata(self, reset_db) -> None:
        video_filepaths = get_video_files()
        base_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        base_t.add_computed_column(metadata=base_t.video.get_metadata())
        validate_update_status(base_t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        result = base_t.where(base_t.metadata.size == 2234371).select(base_t.metadata).collect()['metadata'][0]
        assert result == {
            'bit_exact': False,
            'bit_rate': 967260,
            'size': 2234371,
            'metadata': {
                'encoder': 'Lavf60.16.100',
                'major_brand': 'isom',
                'minor_version': '512',
                'compatible_brands': 'isomiso2avc1mp41',
            },
            'streams': [
                {
                    'type': 'video',
                    'width': 640,
                    'height': 360,
                    'frames': 462,
                    'time_base': 1.0 / 12800,
                    'duration': 236544,
                    'duration_seconds': 236544.0 / 12800,
                    'average_rate': 25.0,
                    'base_rate': 25.0,
                    'guessed_rate': 25.0,
                    'metadata': {
                        'language': 'und',
                        'handler_name': 'L-SMASH Video Handler',
                        'vendor_id': '[0][0][0][0]',
                        'encoder': 'Lavc60.31.102 libx264',
                    },
                    'codec_context': {'name': 'h264', 'codec_tag': 'avc1', 'profile': 'High', 'pix_fmt': 'yuv420p'},
                }
            ],
        }
        # Test a video with an audio stream and a bunch of other edge cases
        result = base_t.where(base_t.metadata.size == 980192).select(base_t.metadata).collect()['metadata'][0]
        assert result == {
            'bit_exact': False,
            'bit_rate': 521864,
            'size': 980192,
            'metadata': {'ENCODER': 'Lavf60.16.100'},
            'streams': [
                {
                    'type': 'video',
                    'duration': None,
                    'time_base': 0.001,
                    'duration_seconds': None,
                    'frames': 0,
                    'metadata': {
                        'language': 'eng',
                        'ENCODER': 'Lavc60.31.102 libvpx-vp9',
                        'DURATION': '00:00:14.981000000',
                    },
                    'average_rate': 30000.0 / 1001,
                    'base_rate': 30000.0 / 1001,
                    'guessed_rate': 30000.0 / 1001,
                    'width': 640,
                    'height': 360,
                    'codec_context': {
                        'name': 'vp9',
                        'codec_tag': '\\x00\\x00\\x00\\x00',
                        'profile': 'Profile 0',
                        'pix_fmt': 'yuv420p',
                    },
                },
                {
                    'type': 'audio',
                    'duration': None,
                    'time_base': 0.001,
                    'duration_seconds': None,
                    'frames': 0,
                    'metadata': {
                        'language': 'eng',
                        'ENCODER': 'Lavc60.31.102 libopus',
                        'DURATION': '00:00:15.026000000',
                    },
                    'codec_context': {
                        'name': 'opus',
                        'codec_tag': '\\x00\\x00\\x00\\x00',
                        'profile': None,
                        'channels': 2,
                    },
                },
            ],
        }

    # window function that simply passes through the frame
    @pxt.uda(requires_order_by=True, allows_std_agg=False, allows_window=True)
    class agg_fn(pxt.Aggregator):
        def __init__(self):
            self.img = None

        def update(self, frame: PIL.Image.Image) -> None:
            self.img = frame

        def value(self) -> PIL.Image.Image:
            return self.img

    def test_make_video(self, reset_db) -> None:
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls()
        base_t.insert({'video': p} for p in video_filepaths)
        # reference to the frame col requires ordering by base, pos
        from pixeltable.functions.video import make_video

        _ = view_t.select(make_video(view_t.pos, view_t.frame)).group_by(base_t).show()
        # the same without frame col
        view_t.add_computed_column(transformed=view_t.frame.rotate(30), stored=True)
        _ = view_t.select(make_video(view_t.pos, view_t.transformed)).group_by(base_t).show()

        with pytest.raises(excs.Error):
            # make_video() doesn't allow windows
            _ = view_t.select(make_video(view_t.pos, view_t.frame, group_by=base_t)).show()
        with pytest.raises(excs.Error):
            # make_video() requires ordering
            _ = view_t.select(make_video(view_t.frame, order_by=view_t.pos)).show()
        with pytest.raises(excs.Error):
            # incompatible ordering requirements
            _ = (
                view_t.select(make_video(view_t.pos, view_t.frame), make_video(view_t.pos - 1, view_t.transformed))
                .group_by(base_t)
                .show()
            )

        # make sure it works
        _ = view_t.select(self.agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()
        status = view_t.add_computed_column(agg=self.agg_fn(view_t.pos, view_t.frame, group_by=base_t))
        assert status.num_excs == 0
        _ = view_t.select(make_video(view_t.pos, view_t.agg)).group_by(base_t).show()

        # image cols computed with a window function currently need to be stored
        with pytest.raises(excs.Error):
            view_t.add_computed_column(agg2=self.agg_fn(view_t.pos, view_t.frame, group_by=base_t), stored=False)

        # reload from store
        reload_catalog()
        base_t, view_t = pxt.get_table(base_t._name), pxt.get_table(view_t._name)
        _ = view_t.select(self.agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()

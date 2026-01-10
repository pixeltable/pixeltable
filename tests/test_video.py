import math
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.env import Env
from pixeltable.functions.video import frame_iterator, video_splitter
from pixeltable.utils.object_stores import ObjectOps

from .utils import (
    generate_test_video,
    get_audio_files,
    get_video_files,
    reload_catalog,
    skip_test_if_not_installed,
    validate_update_status,
)


class TestVideo:
    def create_tbls(
        self, base_name: str = 'video_tbl', view_name: str = 'frame_view', all_frame_attrs: bool = True
    ) -> tuple[pxt.Table, pxt.Table]:
        pxt.drop_table(view_name, if_not_exists='ignore')
        pxt.drop_table(base_name, if_not_exists='ignore')
        base_t = pxt.create_table(base_name, {'video': pxt.Video})
        view_t = pxt.create_view(
            view_name, base_t, iterator=frame_iterator(base_t.video, fps=1, all_frame_attrs=all_frame_attrs)
        )
        return base_t, view_t

    def create_and_insert(self, stored: bool | None, paths: list[str]) -> tuple[pxt.Table, pxt.Table]:
        base_t, view_t = self.create_tbls()
        _ = ObjectOps.count(view_t._id, default_output_dest=True)

        view_t.add_computed_column(transform=view_t.frame.rotate(90), stored=stored)
        base_t.insert({'video': p} for p in paths)
        total_num_rows = view_t.count()
        # TODO: uncomment when we support to_sql_expr() for JsonPathExpr
        # num_key_frames = view_t.where(view_t.frame_attrs.key_frame.astype(pxt.Bool)).count()
        # assert num_key_frames > 0
        frame_attrs = view_t.where(view_t.pos == 0).select(view_t.frame_attrs).collect()[0, 0]
        assert isinstance(frame_attrs['key_frame'], bool) and frame_attrs['key_frame']
        result = (
            view_t.where(view_t.pos >= 5)
            .select(view_t.video, view_t.frame_attrs['index'], view_t.frame, view_t.transform)
            .collect()
        )
        assert len(result) == total_num_rows - len(paths) * 5
        result = view_t.select(view_t.frame, view_t.transform).show(3)
        assert len(result) == 3
        result = view_t.select(view_t.frame, view_t.transform).collect()
        assert len(result) == total_num_rows
        # Try inserting a row with a `None` video; confirm that it produces no additional rows in the view
        base_t.insert(video=None)
        result = view_t.select(view_t.frame, view_t.transform).collect()
        assert len(result) == total_num_rows
        return base_t, view_t

    def test_basic(self, reset_db: None) -> None:
        video_filepaths = get_video_files()

        # computed images are not stored
        _, view = self.create_and_insert(False, video_filepaths)
        assert ObjectOps.count(view._id, default_output_dest=True) == 0

        # computed images are stored
        tbl, view = self.create_and_insert(True, video_filepaths)
        assert ObjectOps.count(view._id, default_output_dest=True) == view.count()

        # revert() also removes computed images
        tbl.insert({'video': p} for p in video_filepaths)
        tbl.revert()
        assert ObjectOps.count(view._id, default_output_dest=True) == view.count()

    def test_query(self, reset_db: None) -> None:
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

    # Bug(PXT-842): pxt.create_view('invalid_args'..., actually creates a corrupted view in the catalog (despite raising
    # an error). That view causes a failure during the database validation.
    @pytest.mark.corrupts_db
    def test_fps(self, reset_db: None) -> None:
        path = get_video_files()[0]
        videos = pxt.create_table('videos', {'video': pxt.Video})
        frames_all = pxt.create_view('frames_all', videos, iterator=frame_iterator(videos.video))
        frames_1_0 = pxt.create_view('frames_1_0', videos, iterator=frame_iterator(videos.video, fps=1))
        frames_0_5 = pxt.create_view('frames_0_5', videos, iterator=frame_iterator(videos.video, fps=1 / 2))
        frames_0_33 = pxt.create_view('frames_0_33', videos, iterator=frame_iterator(videos.video, fps=1 / 3))
        frames_1000 = pxt.create_view('frames_1000', videos, iterator=frame_iterator(videos.video, fps=1000))
        num_frames_10 = pxt.create_view('num_frames_10', videos, iterator=frame_iterator(videos.video, num_frames=10))
        num_frames_50 = pxt.create_view('num_frames_50', videos, iterator=frame_iterator(videos.video, num_frames=50))
        num_frames_1000 = pxt.create_view(
            'num_frames_1000', videos, iterator=frame_iterator(videos.video, num_frames=1000)
        )
        videos.insert(video=path)
        assert frames_all.count() == 448
        assert frames_1_0.count() == 15
        assert frames_0_5.count() == 8
        assert frames_0_33.count() == 5
        assert frames_1000.count() == 448
        assert num_frames_10.count() == 10
        assert num_frames_50.count() == 50
        assert num_frames_1000.count() == 448

        with pytest.raises(pxt.Error, match='At most one of'):
            _ = pxt.create_view('invalid_args', videos, iterator=frame_iterator(videos.video, fps=1 / 2, num_frames=10))

    def test_frame_iterator_seek(self, reset_db: None) -> None:
        """
        Test that we can seek to specific frames in the video iterator and get consistent results.

        Loads the first 50 frames of a video with various fps and num_frames settings, then queries for frames at
        specific positions and checks that the output is pixel-identical.

        The test runs against both the fixed-framerate and variable-framerate versions of the test video.
        """
        paths = [p for p in get_video_files() if '10-Second Video' in p]
        assert len(paths) >= 2
        for p in paths:
            for kwargs in (
                {'fps': None},
                {'fps': 1},
                {'fps': 0.5},
                {'fps': 1000},
                {'num_frames': 10},
                {'num_frames': 50},
                {'num_frames': 10000},
            ):
                videos = pxt.create_table('videos', {'video': pxt.Video}, if_exists='replace_force')
                view = pxt.create_view('frames', videos, iterator=frame_iterator(videos.video, **kwargs))
                videos.insert(video=p)
                # Load the first 20 frames sequentially
                frames = view.select(view.frame).where(view.pos < 20).order_by(view.frame).collect()['frame']
                # Now load them one at a time (we intentionally do this in separate queries)
                for pos in (3, 7, 11, 15):
                    res = view.where(view.pos == pos).select(view.frame).collect()['frame']
                    if len(res) == 0:
                        assert len(frames) <= pos
                    else:
                        selected_frame = res[0]
                        assert isinstance(selected_frame, PIL.Image.Image)
                        # Ensure we get the bitmap-identical frame
                        assert selected_frame == frames[pos]

    # Bug(PXT-842): pxt.create_view('invalid'..., actually creates a corrupted view in the catalog (despite raising
    # an error). That view causes a failure during the database validation.
    @pytest.mark.corrupts_db
    def test_keyframes_only(self, reset_db: None) -> None:
        path = get_video_files()[0]
        videos = pxt.create_table('videos', {'video': pxt.Video})

        # Test keyframes_only=True extracts all keyframes
        keyframes = pxt.create_view(
            'keyframes', videos, iterator=frame_iterator(videos.video, keyframes_only=True, all_frame_attrs=True)
        )
        frames = pxt.create_view('frames', videos, iterator=frame_iterator(videos.video, fps=0, all_frame_attrs=True))

        videos.insert(video=path)

        # Verify keyframes were extracted
        keyframes_count = keyframes.count()
        res = frames.order_by(frames.pos).collect()
        assert keyframes_count == sum(attrs['key_frame'] for attrs in res['frame_attrs'])

        with pytest.raises(pxt.Error, match='At most one of'):
            _ = pxt.create_view('invalid', videos, iterator=frame_iterator(videos.video, keyframes_only=True, fps=1))

        with pytest.raises(pxt.Error, match='At most one of'):
            _ = pxt.create_view(
                'invalid', videos, iterator=frame_iterator(videos.video, keyframes_only=True, num_frames=10)
            )

    def test_computed_cols(self, reset_db: None) -> None:
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls()
        base_t.insert({'video': p} for p in video_filepaths)
        _ = view_t.select(view_t.video, view_t.frame).collect()
        # c2 and c4 depend directly on c1, c3 depends on it indirectly
        view_t.add_computed_column(c1=view_t.frame.resize([224, 224]))
        view_t.add_computed_column(c2=view_t.c1.rotate(10))
        view_t.add_computed_column(c3=view_t.c2.rotate(20))
        view_t.add_computed_column(c4=view_t.c1.rotate(30))
        for name in ['c1', 'c2', 'c3', 'c4']:
            assert view_t._tbl_version_path.tbl_version.get().cols_by_name[name].is_stored
        base_t.insert({'video': p} for p in video_filepaths)
        _ = view_t.select(view_t.frame, view_t.c1, view_t.c2, view_t.c3, view_t.c4).collect()

    def test_frame_attrs(self, reset_db: None) -> None:
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls(all_frame_attrs=True)
        base_t.insert([{'video': video_filepaths[0]}])
        all_attrs = set(view_t.limit(1).select(view_t.frame_attrs).collect()[0, 0].keys())
        assert all_attrs == {'index', 'pts', 'dts', 'time', 'is_corrupt', 'key_frame', 'pict_type', 'interlaced_frame'}
        _, view_t = self.create_tbls(all_frame_attrs=False)
        default_attrs = set(view_t.get_metadata()['columns'].keys())
        assert default_attrs == {'frame', 'pos', 'frame_idx', 'pos_msec', 'pos_frame', 'video'}

    def test_get_metadata(self, reset_db: None) -> None:
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
        img: PIL.Image.Image | None

        def __init__(self) -> None:
            self.img = None

        def update(self, frame: PIL.Image.Image) -> None:
            self.img = frame

        def value(self) -> PIL.Image.Image:
            return self.img

    def test_make_video(self, reset_db: None) -> None:
        video_filepaths = get_video_files()
        base_t, view_t = self.create_tbls()
        base_t.insert({'video': p} for p in video_filepaths)
        # reference to the frame col requires ordering by base, pos
        from pixeltable.functions.video import make_video

        _ = view_t.select(view_t.pos, view_t.frame).show()
        _ = view_t.select(make_video(view_t.pos, view_t.frame)).group_by(base_t).show()
        # the same without frame col
        view_t.add_computed_column(transformed=view_t.frame.rotate(30), stored=True)
        _ = view_t.select(make_video(view_t.pos, view_t.transformed)).group_by(base_t).show()

        with pytest.raises(pxt.Error):
            # make_video() doesn't allow windows
            _ = view_t.select(make_video(view_t.pos, view_t.frame, group_by=base_t)).show()
        with pytest.raises(pxt.Error):
            # make_video() requires ordering
            _ = view_t.select(make_video(view_t.frame, order_by=view_t.pos)).show()
        with pytest.raises(pxt.Error):
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
        with pytest.raises(pxt.Error):
            view_t.add_computed_column(agg2=self.agg_fn(view_t.pos, view_t.frame, group_by=base_t), stored=False)

        # reload from store
        reload_catalog()
        base_t, view_t = pxt.get_table(base_t._name), pxt.get_table(view_t._name)
        _ = view_t.select(self.agg_fn(view_t.pos, view_t.frame, group_by=base_t)).show()

    @pytest.mark.parametrize('mode', ['fast', 'accurate'])
    def test_clip(self, mode: Literal['fast', 'accurate'], reset_db: None) -> None:
        t = pxt.create_table('get_clip_test', {'video': pxt.Video}, media_validation='on_write')
        # TODO: this test is not working with the VFR sample video.
        video_filepaths = get_video_files(include_vfr=False)
        t.insert({'video': p} for p in video_filepaths)

        clip_5_10 = t.video.clip(start_time=5.0, end_time=10.0, mode=mode)
        clip_0_5 = t.video.clip(start_time=0.0, duration=5.0, mode=mode)
        clip_10_end = t.video.clip(start_time=10.0, mode=mode)
        result = t.select(
            clip_5_10=clip_5_10,
            clip_5_10_duration=clip_5_10.get_metadata().streams[0].duration_seconds,
            clip_0_5=clip_0_5,
            clip_0_5_duration=clip_0_5.get_metadata().streams[0].duration_seconds,
            clip_10_end=clip_10_end,
        ).collect()
        assert len(result) == len(video_filepaths)
        df = result.to_pandas()
        assert df['clip_5_10'].notnull().all()
        assert df['clip_0_5'].notnull().all()
        assert df['clip_10_end'].notnull().all()
        assert df['clip_5_10_duration'].between(5.0, 6.0).all()
        assert df['clip_0_5_duration'].between(5.0, 6.0).all()

        # insert generated clips into video_t to verify that they are valid videos
        t.insert({'video': row['clip_5_10']} for row in result)
        t.insert({'video': row['clip_0_5']} for row in result)
        t.insert({'video': row['clip_10_end']} for row in result)

        # requesting a time range past the end of the video returns None
        duration = t.video.get_metadata().streams[0].duration_seconds
        result_df = (
            t.where(duration != None).select(clip=t.video.clip(start_time=1000.0, mode=mode)).collect().to_pandas()
        )
        assert result_df['clip'].isnull().all(), result_df['clip']

        with pytest.raises(pxt.Error, match='start_time must be non-negative'):
            _ = t.select(invalid_clip=t.video.clip(start_time=-1.0)).collect()

        with pytest.raises(pxt.Error, match=r'end_time \(5.0\) must be greater than start_time \(10.0\)'):
            _ = t.select(invalid_clip=t.video.clip(start_time=10.0, end_time=5.0)).collect()

        with pytest.raises(pxt.Error, match='duration must be positive'):
            _ = t.select(invalid_clip=t.video.clip(start_time=10.0, duration=-1.0)).collect()

        with pytest.raises(pxt.Error, match='end_time and duration cannot both be specified'):
            _ = t.select(invalid_clip=t.video.clip(start_time=10.0, end_time=20.0, duration=10.0)).collect()

    def test_extract_frame(self, reset_db: None) -> None:
        video_filepaths = get_video_files()
        t = pxt.create_table('video_tbl', {'video': pxt.Video})
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))

        status = t.add_computed_column(frame_at_1s=t.video.extract_frame(timestamp=1.0))
        assert status.num_excs == 0
        status = t.add_computed_column(
            frame_at_minus_1s=t.video.extract_frame(timestamp=t.video.get_metadata().streams[0].duration_seconds - 1.0)
        )
        assert status.num_excs == 0
        _ = t.select(t.video.get_metadata()).collect()
        result = t.select(
            width=t.video.get_metadata().streams[0].width,
            height=t.video.get_metadata().streams[0].height,
            at_1s_width=t.frame_at_1s.width,
            at_1s_height=t.frame_at_1s.height,
        ).collect()
        assert len(result) == len(video_filepaths)
        result_df = result.to_pandas()
        assert result_df['width'].eq(result_df['at_1s_width']).all()
        assert result_df['height'].eq(result_df['at_1s_height']).all()

        # get frame close to the end of the video
        result = (
            t.where(t.video.get_metadata().streams[0].duration_seconds != None)
            .select(
                width=t.video.get_metadata().streams[0].width,
                height=t.video.get_metadata().streams[0].height,
                at_minus_1s_width=t.frame_at_minus_1s.width,
                at_minus_1s_height=t.frame_at_minus_1s.height,
            )
            .collect()
        )
        result_df = result.to_pandas()
        assert result_df['width'].eq(result_df['at_minus_1s_width']).all()
        assert result_df['height'].eq(result_df['at_minus_1s_height']).all()

        # get frame past the end of the video
        result_df = t.select(frame=t.video.extract_frame(timestamp=1000.0)).collect().to_pandas()
        assert result_df['frame'].isnull().all()

        with pytest.raises(pxt.Error):
            t.add_computed_column(invalid3=t.video.extract_frame(timestamp=-1.0))

    def _validate_segments(
        self,
        segments: list[str],
        total_duration: float,
        duration: float | None = None,
        durations: list[float] | None = None,
        eps: float | None = None,
    ) -> None:
        assert duration is None or durations is None
        t = pxt.create_table('validate_segments', {'segment': pxt.Video}, media_validation='on_write')
        t.insert({'segment': s} for s in segments)
        duration_expr = t.segment.get_metadata().streams[0].duration_seconds
        result = t.select(duration=duration_expr).head(n=len(segments))  # make sure output is ordered chronologically
        assert len(result) == len(segments)
        assert sum(result['duration']) == pytest.approx(total_duration, abs=0.01)
        if duration is not None:
            df = result.to_pandas()
            # :-1: omit last row since it typically contains the remainder and may be shorter
            assert df.iloc[:-1]['duration'].between(duration - eps, duration + eps).all()
        if durations is not None:
            # strict=False: we don't know how many segments got created
            assert all(
                expected - eps <= actual and expected + eps >= actual
                for expected, actual in zip(durations, result['duration'][:-1], strict=False)
            )
        pxt.drop_table('validate_segments')

    @pytest.mark.parametrize('mode', ['fast', 'accurate'])
    def test_segment_video_duration(self, mode: Literal['fast', 'accurate'], reset_db: None) -> None:
        t = pxt.create_table('test_segments', {'video': pxt.Video})
        # TODO: this test is not working with the VFR sample video.
        t.insert({'video': f} for f in get_video_files(include_vfr=False))

        duration = t.video.get_metadata().streams[0].duration_seconds
        result = (
            t.where(duration != None)
            .select(duration=duration, segments=t.video.segment_video(duration=3.0, mode=mode))
            .collect()
        )
        eps = 1.0 if mode == 'fast' else 0.0
        for row in result:
            total_duration = row['duration']
            segments = row['segments']
            assert len(segments) >= 1
            self._validate_segments(segments, total_duration, duration=3.0, eps=eps)

        # split at midpoint
        result = (
            t.where(duration != None)
            .select(duration=duration, segments=t.video.segment_video(duration=duration / 2 + 0.1, mode=mode))
            .collect()
        )
        for row in result:
            total_duration = row['duration']
            segments = row['segments']
            assert len(segments) >= 1
            self._validate_segments(segments, total_duration)

    @pytest.mark.parametrize('mode', ['fast', 'accurate'])
    def test_segment_video_segment_times(self, mode: Literal['fast', 'accurate'], reset_db: None) -> None:
        t = pxt.create_table('test_segments', {'video': pxt.Video})
        t.insert([{'video': f} for f in get_video_files()])

        duration = t.video.get_metadata().streams[0].duration_seconds
        segment_times = [6.0, 11.0, 16.0]
        result = (
            t.where(duration != None)
            .select(duration=duration, segments=t.video.segment_video(segment_times=segment_times, mode=mode))
            .collect()
        )
        start_times = [0.0, *segment_times]
        durations = [start_times[i + 1] - start_times[i] for i in range(len(start_times) - 1)]
        for row in result:
            total_duration = row['duration']
            segments = row['segments']
            assert len(segments) >= 1
            eps = 1.0 if mode == 'fast' else 0.0
            self._validate_segments(segments, total_duration, durations=durations, eps=eps)

    def test_segment_video_errors(self, reset_db: None) -> None:
        t = pxt.create_table('test_segments', {'video': pxt.Video})
        t.insert([{'video': f} for f in get_video_files()])

        with pytest.raises(pxt.Error, match='duration must be positive'):
            _ = t.select(invalid=t.video.segment_video(duration=0.0)).collect()

        with pytest.raises(pxt.Error, match='video_encoder is not supported'):
            _ = t.select(invalid=t.video.segment_video(duration=5.0, mode='fast', video_encoder='libx264')).collect()

        with pytest.raises(pxt.Error, match='video_encoder_args is not supported'):
            _ = t.select(
                invalid=t.video.segment_video(duration=5.0, mode='fast', video_encoder_args={'crf': 18})
            ).collect()

        with pytest.raises(pxt.Error, match='segment_times cannot be empty'):
            _ = t.select(invalid=t.video.segment_video(segment_times=[])).collect()

        with pytest.raises(pxt.Error, match='duration and segment_times cannot both be specified'):
            _ = t.select(invalid=t.video.segment_video(duration=1.0, segment_times=[1.0, 2.0])).collect()

    def test_concat_videos(self, reset_db: None) -> None:
        video_filepaths = get_video_files()[:3]  # Use first 3 videos
        from pixeltable.functions.video import concat_videos

        t = pxt.create_table('concat_videos_test', {'video': pxt.Video})
        t.insert({'video': p} for p in video_filepaths)

        # basic test: reassemble segments into original video
        t.add_computed_column(segments=t.video.segment_video(duration=5.0))
        t.add_computed_column(concat=concat_videos(t.segments))
        res_df = (
            t.select(
                url=t.video.fileurl,
                segments=t.segments,
                duration=t.video.get_duration(),
                concat_duration=t.concat.get_duration(),
            )
            .collect()
            .to_pandas()
        )
        print(res_df)
        assert res_df['duration'].between(res_df['concat_duration'] - 0.1, res_df['concat_duration'] + 0.1).all()

        # assemble videos of different origin into a single video
        u = pxt.create_table('concat_videos_test2', {'v1': pxt.Video, 'v2': pxt.Video, 'v3': pxt.Video})
        u.insert([{'v1': video_filepaths[0], 'v2': video_filepaths[1], 'v3': video_filepaths[2]}])
        status = u.add_computed_column(
            concat=concat_videos([u.v1.astype(pxt.String), u.v2.astype(pxt.String), u.v3.astype(pxt.String)])
        )
        assert status.num_excs == 0
        res = u.select(
            u.v1.get_metadata().streams[0].duration_seconds,
            u.v2.get_metadata().streams[0].duration_seconds,
            u.v3.get_metadata().streams[0].duration_seconds,
            u.concat.get_metadata().streams[0].duration_seconds,
        ).collect()
        # Verify all videos were concatenated
        durations = res.to_pandas().iloc[0]
        concat_duration = durations.iloc[3]
        assert concat_duration is not None

    def test_concat_videos_mixed_formats(self, reset_db: None, tmp_path: Path) -> None:
        from pixeltable.functions.video import concat_videos

        # mixed audio
        no_audio = generate_test_video(tmp_path, duration=1.0, has_audio=False)
        with_audio = generate_test_video(tmp_path, duration=1.5, has_audio=True)

        t = pxt.create_table('test_mixed_audio', {'v1': pxt.Video, 'v2': pxt.Video, 'v3': pxt.Video})
        t.insert([{'v1': no_audio, 'v2': with_audio, 'v3': no_audio}])
        status = t.add_computed_column(
            concat=concat_videos([t.v1.astype(pxt.String), t.v2.astype(pxt.String), t.v3.astype(pxt.String)])
        )
        assert status.num_excs == 0
        res = t.select(t.concat.get_metadata().streams[0].duration_seconds).collect()
        duration = res[0, 0]
        assert abs(duration - (1.0 + 1.5 + 1.0)) < 0.1

        short_video = generate_test_video(tmp_path, duration=0.2, has_audio=False)
        yuv422_video = generate_test_video(tmp_path, duration=0.5, pix_fmt='yuv422p', has_audio=False)

        t = pxt.create_table('test_edge_cases', {'v1': pxt.Video, 'v2': pxt.Video, 'v3': pxt.Video})
        t.insert([{'v1': short_video, 'v2': yuv422_video, 'v3': no_audio}])
        status = t.add_computed_column(
            concat=concat_videos([t.v1.astype(pxt.String), t.v2.astype(pxt.String), t.v3.astype(pxt.String)])
        )
        assert status.num_excs == 0
        # verify that we got a video
        res = t.select(fileurl=t.concat.fileurl, md=t.concat.get_metadata()).collect()
        assert len(res) == 1
        assert res[0]['fileurl'] is not None
        assert res[0]['md']['streams'][0]['type'] == 'video'

        # error case: mixed resolutions
        low_res = generate_test_video(tmp_path, duration=0.5, size='176x144', has_audio=False)
        high_res = generate_test_video(tmp_path, duration=0.5, size='1920x1080', has_audio=False)
        mid_res = generate_test_video(tmp_path, duration=0.5, size='640x360', has_audio=False)

        t = pxt.create_table('test_resolution', {'v1': pxt.Video, 'v2': pxt.Video, 'v3': pxt.Video})
        t.insert([{'v1': low_res, 'v2': high_res, 'v3': mid_res}])
        with pytest.raises(pxt.Error, match='requires that all videos have the same resolution'):
            _ = t.add_computed_column(
                concat=concat_videos([t.v1.astype(pxt.String), t.v2.astype(pxt.String), t.v3.astype(pxt.String)])
            )

    def _validate_splitter_segments(
        self,
        base: pxt.Table,
        segments_view: pxt.Table,
        overlap: float | None,
        min_segment_duration: float,
        expected_durations: list[float] | None = None,
        eps: float = 0.0,  # epsilon used in pytest.approx()
    ) -> None:
        t = base
        s = segments_view
        if overlap is None:
            overlap = 0.0

        # we cannot directly verify the number of segments, because they can diverge from the target duration;
        res = t.select(t.video, time_base=t.video.get_metadata().streams[0].time_base).collect()
        time_bases = {row['video']: row['time_base'] for row in res}
        res = t.select(t.video, duration=t.video.get_duration()).collect()
        video_durations = {row['video']: row['duration'] for row in res}
        segments_md = (
            s.select(s.video, s.pos, s.segment_start, s.segment_end, s.segment_start_pts, s.segment_end_pts)
            .order_by(s.video, s.pos)
            .collect()
        )
        last_pos: dict[str, int] = {}
        for row in segments_md:
            last_pos[row['video']] = row['pos']

        for i in range(len(segments_md)):
            assert segments_md[i]['segment_end'] >= segments_md[i]['segment_start']
            assert segments_md[i]['segment_end_pts'] >= segments_md[i]['segment_start_pts']
            if segments_md[i]['pos'] > 0:
                # verify segment_end/start are consecutive, minus overlap
                assert segments_md[i]['segment_start'] == pytest.approx(
                    segments_md[i - 1]['segment_end'] - overlap, abs=eps
                )
                assert segments_md[i]['segment_start_pts'] == pytest.approx(
                    segments_md[i - 1]['segment_end_pts'] - round(overlap / time_bases[segments_md[i]['video']]), abs=1
                )
                assert segments_md[i]['segment_end'] - segments_md[i]['segment_start'] >= min_segment_duration

            # compare segment lengths, except for the last one, which can be shorter
            if expected_durations is not None and segments_md[i]['pos'] < last_pos[segments_md[i]['video']]:
                pos = segments_md[i]['pos']
                # abs=eps + 0.01: we're seeing some deviation even for accurate segmentation
                assert segments_md[i]['segment_end'] - segments_md[i]['segment_start'] == pytest.approx(
                    expected_durations[pos], abs=eps + 0.01
                )

            if min_segment_duration == 0.0 and i > 0 and segments_md[i]['pos'] == 0:
                # verify that the last segment's segment_end matches video duration
                # abs=0.1: for some reason, even accurate segmentation doesn't quite match the final length
                assert segments_md[i - 1]['segment_end'] == pytest.approx(
                    video_durations[segments_md[i - 1]['video']], abs=0.1
                )
        if min_segment_duration == 0.0:
            # abs=0.1: for some reason, even accurate segmentation doesn't quite match the final length
            assert segments_md[-1]['segment_end'] == pytest.approx(video_durations[segments_md[-1]['video']], abs=0.1)

        # Verify segments are valid videos by inserting them into a table with validation
        segments = s.select(url=s.video_segment.fileurl).collect()
        if len(segments) > 0:
            validation_t = pxt.create_table('segment_validation', {'segment': pxt.Video}, media_validation='on_write')
            validation_t.insert([{'segment': row['url']} for row in segments], on_error='abort')
            pxt.drop_table('segment_validation')

    @pytest.mark.parametrize(
        'segment_duration,mode',
        [(5.0, 'fast'), (5.0, 'accurate'), (10.0, 'fast'), (10.0, 'accurate'), (100.0, 'fast'), (100.0, 'accurate')],
    )
    def test_video_splitter_duration(
        self, segment_duration: float, mode: Literal['fast', 'accurate'], reset_db: None
    ) -> None:
        video_filepaths = get_video_files()
        overlaps = [0.0, 1.0, 4.0] if mode == 'fast' else [None]
        eps = 0.1 if mode == 'fast' else 0.0
        for min_segment_duration in [0.0, segment_duration]:
            for overlap in overlaps:
                t = pxt.create_table('videos', {'video': pxt.Video})
                t.insert([{'video': p} for p in video_filepaths])
                s = pxt.create_view(
                    'segments',
                    t,
                    iterator=video_splitter(
                        t.video,
                        duration=segment_duration,
                        mode=mode,
                        overlap=overlap,
                        min_segment_duration=min_segment_duration,
                    ),
                )
                self._validate_splitter_segments(t, s, overlap, min_segment_duration, eps=eps)
                pxt.drop_table('videos', force=True)

    @pytest.mark.parametrize('segment_times,mode', [([6.0, 11.0, 16.0], 'fast'), ([6.0, 11.0, 16.0], 'accurate')])
    def test_video_splitter_segment_times(
        self, segment_times: list[float], mode: Literal['fast', 'accurate'], reset_db: None
    ) -> None:
        eps = 0.1 if mode == 'fast' else 0.0
        video_filepaths = get_video_files()
        t = pxt.create_table('videos', {'video': pxt.Video})
        t.insert({'video': p} for p in video_filepaths)
        s = pxt.create_view('segments', t, iterator=video_splitter(t.video, segment_times=segment_times, mode=mode))
        start_times = [0.0, *segment_times]
        durations = [start_times[i + 1] - start_times[i] for i in range(len(start_times) - 1)]
        self._validate_splitter_segments(t, s, 0.0, 0.0, expected_durations=durations, eps=eps)

    @pytest.mark.parametrize('mode', ['fast', 'accurate'])
    def test_video_splitter_empty_segment_times(self, mode: Literal['fast', 'accurate'], reset_db: None) -> None:
        video_filepaths = get_video_files()
        t = pxt.create_table('videos', {'video': pxt.Video})
        t.insert({'video': p} for p in video_filepaths)
        v = pxt.create_view('s', t, iterator=video_splitter(t.video, segment_times=[], mode=mode))
        res = v.select(video=v.video.fileurl, segment=v.video_segment.fileurl).collect()
        assert all(row['video'] == row['segment'] for row in res)

    def test_video_splitter_errors(self, reset_db: None) -> None:
        t = pxt.create_table('videos', {'video': pxt.Video})
        with pytest.raises(pxt.Error, match='Must specify either duration or segment_times'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video))
        with pytest.raises(pxt.Error, match='duration must be a positive number'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, duration=-1))
        with pytest.raises(pxt.Error, match='overlap must be less than duration'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, duration=1, overlap=1))
        with pytest.raises(pxt.Error, match='duration must be at least min_segment_duration'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, duration=1, min_segment_duration=2))
        with pytest.raises(pxt.Error, match='Cannot specify overlap'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, duration=10, mode='accurate', overlap=5))
        with pytest.raises(pxt.Error, match='Cannot specify video_encoder'):
            _ = pxt.create_view(
                's', t, iterator=video_splitter(t.video, duration=10, mode='fast', video_encoder='libx264')
            )
        with pytest.raises(pxt.Error, match='Cannot specify video_encoder_args'):
            _ = pxt.create_view(
                's', t, iterator=video_splitter(t.video, duration=10, mode='fast', video_encoder_args={'crf': 18})
            )
        with pytest.raises(pxt.Error, match='duration and segment_times cannot both be specified'):
            _ = pxt.create_view(
                's', t, iterator=video_splitter(t.video, duration=10, segment_times=[1, 2], mode='fast')
            )
        with pytest.raises(pxt.Error, match='Cannot specify overlap'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, mode='accurate', duration=3, overlap=1))
        with pytest.raises(pxt.Error, match='overlap cannot be specified with segment_times'):
            _ = pxt.create_view('s', t, iterator=video_splitter(t.video, segment_times=[1, 2], overlap=1))

    @pytest.mark.skipif(
        os.environ.get('PXTTEST_CI_OS') == 'ubuntu-x64-t4', reason='Fonts not available on t4 CI instances'
    )
    def test_overlay_text(self, reset_db: None, tmp_path: Path) -> None:
        t = pxt.create_table('videos', {'video': pxt.Video})
        t.add_computed_column(clip_5s=t.video.clip(start_time=0, duration=5))

        text = "Line 1\nLine2: 'quoted text'"
        t.add_computed_column(
            o1=t.clip_5s.overlay_text(
                text,
                color='black',
                opacity=0.5,
                horizontal_align='left',
                horizontal_margin=10,
                vertical_align='top',
                box=False,
            )
        )
        t.add_computed_column(
            o2=t.clip_5s.overlay_text(
                text,
                color='red',
                opacity=0.8,
                horizontal_align='center',
                vertical_margin=10,
                vertical_align='bottom',
                box=True,
                box_color='blue',
                box_opacity=0.5,
                box_border=[10, 20, 30, 40],
            )
        )
        t.add_computed_column(
            o3=t.clip_5s.overlay_text(
                text,
                color='yellow',
                opacity=1.0,
                horizontal_align='right',
                vertical_margin=10,
                vertical_align='center',
                box=True,
                box_color='red',
                box_opacity=0.8,
                box_border=[10, 20, 30],
            )
        )
        t.add_computed_column(
            o4=t.clip_5s.overlay_text(
                text,
                color='red',
                opacity=0.8,
                horizontal_align='center',
                vertical_margin=10,
                vertical_align='bottom',
                box=True,
                box_color='blue',
                box_opacity=0.5,
                box_border=[10, 20],
            )
        )
        t.add_computed_column(
            o5=t.clip_5s.overlay_text(
                text,
                color='yellow',
                opacity=1.0,
                horizontal_align='right',
                vertical_margin=10,
                vertical_align='center',
                box=True,
                box_color='red',
                box_opacity=0.8,
                box_border=[10],
            )
        )
        rows = [{'video': v} for v in get_video_files()]
        status = t.insert(rows)
        assert status.num_excs == 0

        # also check the generated drawtext commands
        assert pxtf.video._create_drawtext_params(
            text,
            font=None,
            font_size=24,
            color='black',
            opacity=0.5,
            horizontal_align='left',
            horizontal_margin=10,
            vertical_align='top',
            vertical_margin=0,
            box=False,
            box_color='black',
            box_opacity=1.0,
            box_border=None,
        ) == ["text='Line 1\nLine2\\: \\'quoted text\\''", 'fontsize=24', 'fontcolor=black@0.5', 'x=10', 'y=0']
        assert pxtf.video._create_drawtext_params(
            text,
            font=None,
            font_size=24,
            color='red',
            opacity=0.8,
            horizontal_align='center',
            horizontal_margin=0,
            vertical_align='bottom',
            vertical_margin=10,
            box=True,
            box_color='blue',
            box_opacity=0.5,
            box_border=[10, 20, 30, 40],
        ) == [
            "text='Line 1\nLine2\\: \\'quoted text\\''",
            'fontsize=24',
            'fontcolor=red@0.8',
            'x=(w-text_w)/2',
            'y=h-text_h-10',
            'box=1',
            'boxcolor=blue@0.5',
            'boxborderw=10|20|30|40',
        ]
        assert pxtf.video._create_drawtext_params(
            text,
            font=None,
            font_size=24,
            color='yellow',
            opacity=1.0,
            horizontal_align='right',
            horizontal_margin=0,
            vertical_align='center',
            vertical_margin=10,
            box=True,
            box_color='red',
            box_opacity=0.8,
            box_border=[10, 20, 30],
        ) == [
            "text='Line 1\nLine2\\: \\'quoted text\\''",
            'fontsize=24',
            'fontcolor=yellow',
            'x=w-text_w',
            'y=(h-text_h)/2',
            'box=1',
            'boxcolor=red@0.8',
            'boxborderw=10|20|30',
        ]
        assert pxtf.video._create_drawtext_params(
            text,
            font=None,
            font_size=24,
            color='red',
            opacity=0.8,
            horizontal_align='center',
            horizontal_margin=0,
            vertical_align='bottom',
            vertical_margin=10,
            box=True,
            box_color='blue',
            box_opacity=0.5,
            box_border=[10, 20],
        ) == [
            "text='Line 1\nLine2\\: \\'quoted text\\''",
            'fontsize=24',
            'fontcolor=red@0.8',
            'x=(w-text_w)/2',
            'y=h-text_h-10',
            'box=1',
            'boxcolor=blue@0.5',
            'boxborderw=10|20',
        ]
        assert pxtf.video._create_drawtext_params(
            text,
            font=None,
            font_size=24,
            color='yellow',
            opacity=1.0,
            horizontal_align='right',
            horizontal_margin=0,
            vertical_align='center',
            vertical_margin=10,
            box=True,
            box_color='red',
            box_opacity=0.8,
            box_border=[10],
        ) == [
            "text='Line 1\nLine2\\: \\'quoted text\\''",
            'fontsize=24',
            'fontcolor=yellow',
            'x=w-text_w',
            'y=(h-text_h)/2',
            'box=1',
            'boxcolor=red@0.8',
            'boxborderw=10',
        ]

        # This doesn't work, because ffmpeg might add a few frames, due to re-encoding.
        # TODO: is this worth fixing?
        # # make sure the clips are still the same length
        # res = t.select(
        #     d=t.clip_5s.get_duration(),
        #     d_o1=t.o1.get_duration(),
        #     d_o2=t.o2.get_duration(),
        #     d_o3=t.o3.get_duration(),
        #     d_o4=t.o4.get_duration(),
        #     d_o5=t.o5.get_duration(),
        # ).collect()
        # df = res.to_pandas()
        # assert df['d'].eq(df['d_o1']).all()
        # assert df['d'].eq(df['d_o2']).all()
        # assert df['d'].eq(df['d_o3']).all()
        # assert df['d'].eq(df['d_o4']).all()
        # assert df['d'].eq(df['d_o5']).all()

    def test_overlay_text_errors(self, reset_db: None, tmp_path: Path) -> None:
        import re

        t = pxt.create_table('videos_errors', {'video': pxt.Video})
        t.insert([{'video': v} for v in get_video_files()])

        with pytest.raises(pxt.Error, match='font_size must be positive'):
            t.select(t.video.overlay_text('Test', font_size=0)).collect()

        with pytest.raises(pxt.Error, match='font_size must be positive'):
            t.select(t.video.overlay_text('Test', font_size=-10)).collect()

        with pytest.raises(pxt.Error, match=re.escape('opacity must be between 0.0 and 1.0')):
            t.select(t.video.overlay_text('Test', opacity=-0.1)).collect()

        with pytest.raises(pxt.Error, match=re.escape('opacity must be between 0.0 and 1.0')):
            t.select(t.video.overlay_text('Test', opacity=1.1)).collect()

        with pytest.raises(pxt.Error, match=re.escape('horizontal_margin must be non-negative')):
            t.select(t.video.overlay_text('Test', horizontal_margin=-5)).collect()

        with pytest.raises(pxt.Error, match=re.escape('vertical_margin must be non-negative')):
            t.select(t.video.overlay_text('Test', vertical_margin=-10)).collect()

        with pytest.raises(pxt.Error, match=re.escape('box_opacity must be between 0.0 and 1.0')):
            t.select(t.video.overlay_text('Test', box=True, box_opacity=-0.5)).collect()

        with pytest.raises(pxt.Error, match=re.escape('box_opacity must be between 0.0 and 1.0')):
            t.select(t.video.overlay_text('Test', box=True, box_opacity=2.0)).collect()

        with pytest.raises(pxt.Error, match=re.escape('box_border must be a list or tuple of 1-4 non-negative ints')):
            t.select(t.video.overlay_text('Test', box=True, box_border=[1, 2, 3, 4, 5])).collect()

        with pytest.raises(pxt.Error, match=re.escape('box_border must be a list or tuple of 1-4 non-negative ints')):
            t.select(t.video.overlay_text('Test', box=True, box_border=[-5, 10])).collect()

    def test_with_audio(self, reset_db: None) -> None:
        from pixeltable.functions.video import with_audio

        # TODO: this test is not working with the VFR sample video.
        video_filepaths = get_video_files(include_vfr=False)
        audio_filepaths = get_audio_files()
        num_rows = min(len(video_filepaths), len(audio_filepaths))

        t = pxt.create_table('test_add_audio', {'video': pxt.Video, 'audio': pxt.Audio})
        validate_update_status(
            t.insert({'video': video_filepaths[i], 'audio': audio_filepaths[i]} for i in range(num_rows)),
            expected_rows=num_rows,
        )

        validate_update_status(t.add_computed_column(with_audio=with_audio(t.video, t.audio)))
        result = t.select(reference=t.video.get_duration(), duration=t.with_audio.get_duration()).collect()
        assert len(result) == num_rows
        assert all(row['reference'] == row['duration'] for row in result)

        # test with offsets
        validate_update_status(
            t.add_computed_column(with_offset=with_audio(t.video, t.audio, video_start_time=1.0, audio_start_time=0.5))
        )
        result = t.select(reference=t.video.get_duration() - 1.0, duration=t.with_offset.get_duration()).collect()
        assert all(math.isclose(row['reference'], row['duration'], abs_tol=0.05) for row in result)

        # test with offsets and duration
        validate_update_status(
            t.add_computed_column(
                with_duration=with_audio(
                    t.video, t.audio, video_duration=5.0, video_start_time=1.0, audio_duration=4.0, audio_start_time=0.5
                )
            )
        )
        result = t.select(duration=t.with_duration.get_duration()).collect()
        assert all(math.isclose(5.0, row['duration'], abs_tol=0.25) for row in result)

        # error conditions
        with pytest.raises(pxt.Error, match='video_offset must be non-negative'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, video_start_time=-1.0))
        with pytest.raises(pxt.Error, match='audio_offset must be non-negative'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, audio_start_time=-1.0))
        with pytest.raises(pxt.Error, match='video_duration must be positive'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, video_duration=0.0))
        with pytest.raises(pxt.Error, match='video_duration must be positive'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, video_duration=-1.0))
        with pytest.raises(pxt.Error, match='audio_duration must be positive'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, audio_duration=0.0))
        with pytest.raises(pxt.Error, match='audio_duration must be positive'):
            t.add_computed_column(invalid=with_audio(t.video, t.audio, audio_duration=-1.0))

    def test_scene_detect(self, reset_db: None) -> None:
        skip_test_if_not_installed('scenedetect')
        video_filepaths = get_video_files()

        test_params: list[tuple[pxt.Function, dict[str, Any]]] = [
            (pxtf.video.scene_detect_adaptive, {}),
            (pxtf.video.scene_detect_content, {}),
            (pxtf.video.scene_detect_threshold, {}),
            (pxtf.video.scene_detect_histogram, {}),
            (pxtf.video.scene_detect_hash, {}),
            (
                pxtf.video.scene_detect_adaptive,
                {
                    'adaptive_threshold': 4.0,
                    'min_scene_len': 20,
                    'window_width': 3,
                    'min_content_val': 15.0,
                    'delta_hue': 2.0,
                    'delta_sat': 2.0,
                    'delta_lum': 2.0,
                    'delta_edges': 1.0,
                    'luma_only': True,
                    'kernel_size': None,
                },
            ),
            (
                pxtf.video.scene_detect_content,
                {
                    'threshold': 27.0,
                    'min_scene_len': 15,
                    'delta_hue': 2.0,
                    'delta_sat': 2.0,
                    'delta_lum': 2.0,
                    'delta_edges': 1.0,
                    'luma_only': True,
                    'kernel_size': None,
                    'filter_mode': 'suppress',
                },
            ),
            (
                pxtf.video.scene_detect_threshold,
                {
                    'threshold': 12.0,
                    'min_scene_len': 15,
                    'fade_bias': 1.0,
                    'add_final_scene': True,
                    'method': 'ceiling',
                },
            ),
            (pxtf.video.scene_detect_histogram, {'threshold': 0.1, 'bins': 256, 'min_scene_len': 15}),
            (pxtf.video.scene_detect_hash, {'threshold': 0.595, 'size': 24, 'lowpass': 3, 'min_scene_len': 15}),
        ]
        for udf, params in test_params:
            t = pxt.create_table('videos', {'video': pxt.Video}, if_exists='replace')
            t.insert({'video': p} for p in video_filepaths)
            status = t.add_computed_column(scenes=udf(t.video, fps=2.0, **params))
            assert status.num_excs == 0
            res = t.select(t.scenes).collect()
            assert len(res) == len(video_filepaths)
            assert all(len(row['scenes']) > 0 for row in res)

        # make sure the output is usable for the VideoSplitter
        v = pxt.create_view(
            'scenes_view',
            t,
            iterator=video_splitter(t.video, segment_times=t.scenes[1:].start_time, mode='accurate'),  # type: ignore[arg-type]
        )
        _ = v.collect()

    def test_default_video_codec(self, reset_db: None) -> None:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=False)
        print(f'ffmpeg -version:\n{result.stdout}')

        default_encoder = Env.get().default_video_encoder
        assert default_encoder == 'libx264'

import glob
import logging
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator, Literal

import av
import pandas as pd
import PIL.Image
from av.container import InputContainer
from deprecated import deprecated

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.utils.local_store import TempStore

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class FrameIterator(ComponentIterator):
    """
    Iterator over frames of a video. At most one of `fps`, `num_frames`, or `keyframes_only` may be specified. If `fps`
    is specified, then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified,
    then the exact number of frames will be extracted. If neither is specified, then all frames will be extracted.

    If `fps` or `num_frames` is large enough to exceed the native framerate of the video, then all frames will be
    extracted. (Frames will never be duplicated; the maximum number of frames extracted is the total number of frames
    in the video.)

    Args:
        fps: Number of frames to extract per second of video. This may be a fractional value, such as `0.5` (one frame
            per two seconds). The first frame of the video will always be extracted.
        num_frames: Exact number of frames to extract. The frames will be spaced as evenly as possible: the video will
            be divided into `num_frames` evenly spaced intervals, and the midpoint of each interval will be used for
            frame extraction.
        keyframes_only: If True, only extract keyframes.
        all_frame_attrs:
            If True, outputs a `pxt.Json` column `frame_attrs` with the following `pyav`-provided attributes
            (for more information, see `pyav`'s documentation on
            [VideoFrame](https://pyav.org/docs/develop/api/video.html#module-av.video.frame) and
            [Frame](https://pyav.org/docs/develop/api/frame.html)):

            * `index` (`int`)
            * `pts` (`int | None`)
            * `dts` (`int | None`)
            * `time` (`float | None`)
            * `is_corrupt` (`bool`)
            * `key_frame` (`bool`)
            * `pict_type` (`int`)
            * `interlaced_frame` (`bool`)

            If False, only outputs frame attributes `frame_idx`, `pos_msec`, and `pos_frame` as separate columns.
    """

    # Input parameters
    video_path: Path
    fps: float | None
    num_frames: int | None
    keyframes_only: bool
    all_frame_attrs: bool

    # Video info
    container: InputContainer
    video_time_base: Fraction
    video_start_time: float
    video_duration: float | None

    # extraction info
    extraction_step: float | None
    next_extraction_time: float | None

    # state
    pos: int
    video_idx: int
    cur_frame: av.VideoFrame | None

    def __init__(
        self,
        video: str,
        *,
        fps: float | None = None,
        num_frames: int | None = None,
        keyframes_only: bool = False,
        all_frame_attrs: bool = False,
    ):
        if int(fps is not None) + int(num_frames is not None) + int(keyframes_only) > 1:
            raise excs.Error('At most one of `fps`, `num_frames` or `keyframes_only` may be specified')

        if fps is not None and fps < 0.0:
            raise excs.Error('`fps` must be a non-negative number')

        if fps == 0.0:
            fps = None  # treat 0.0 as unspecified

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.fps = fps
        self.num_frames = num_frames
        self.keyframes_only = keyframes_only
        self.all_frame_attrs = all_frame_attrs

        self.video_time_base = self.container.streams.video[0].time_base

        start_time = self.container.streams.video[0].start_time or 0
        self.video_start_time = float(start_time * self.video_time_base)

        duration_pts: int | None = self.container.streams.video[0].duration
        if duration_pts is not None:
            self.video_duration = float(duration_pts * self.video_time_base)
        else:
            # As a backup, try to calculate duration from DURATION metadata field
            metadata = self.container.streams.video[0].metadata
            duration_field = metadata.get('DURATION')  # A string like "00:01:23"
            if duration_field is not None:
                assert isinstance(duration_field, str)
                self.video_duration = pd.to_timedelta(duration_field).total_seconds()
            else:
                # TODO: Anything we can do here? Other methods of determining the duration are expensive and
                #     not so appropriate for an iterator initializer.
                self.video_duration = None

        if self.video_duration is None and self.num_frames is not None:
            raise excs.Error(f'Could not determine duration of video: {video}')

        # If self.fps or self.num_frames is specified, we cannot rely on knowing in advance which frame positions will
        # be needed, since for variable framerate videos we do not know in advance the precise timestamp of each frame.
        # The strategy is: predetermine a list of "extraction times", the idealized timestamps of the frames we want to
        # materialize. As we later iterate through the frames, we will choose the frames that are closest to these
        # idealized timestamps.

        self.pos = 0
        self.video_idx = 0
        if self.num_frames is not None:
            # Divide the video duration into num_frames evenly spaced intervals. The extraction times are the midpoints
            # of those intervals.
            self.extraction_step = (self.video_duration - self.video_start_time) / self.num_frames
            self.next_extraction_time = self.video_start_time + self.extraction_step / 2
        elif self.fps is not None:
            self.extraction_step = 1 / self.fps
            self.next_extraction_time = self.video_start_time
        else:
            self.extraction_step = None
            self.next_extraction_time = None

        _logger.debug(
            f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames} '
            f'keyframes_only={self.keyframes_only}'
        )
        self.cur_frame = self.next_frame()

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'video': ts.VideoType(nullable=False),
            'fps': ts.FloatType(nullable=True),
            'num_frames': ts.IntType(nullable=True),
            'keyframes_only': ts.BoolType(nullable=False),
            'all_frame_attrs': ts.BoolType(nullable=False),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        attrs: dict[str, ts.ColumnType]
        fps = kwargs.get('fps')
        if fps is not None and (not isinstance(fps, (int, float)) or fps < 0.0):
            raise excs.Error('`fps` must be a non-negative number')

        if kwargs.get('all_frame_attrs'):
            attrs = {'frame_attrs': ts.JsonType()}
        else:
            attrs = {'frame_idx': ts.IntType(), 'pos_msec': ts.FloatType(), 'pos_frame': ts.IntType()}
        return {**attrs, 'frame': ts.ImageType()}, ['frame']

    def next_frame(self) -> av.VideoFrame | None:
        try:
            return next(self.container.decode(video=0))
        except EOFError:
            return None

    def __next__(self) -> dict[str, Any]:
        while True:
            if self.cur_frame is None:
                raise StopIteration

            next_frame = self.next_frame()

            if self.keyframes_only and not self.cur_frame.key_frame:
                self.cur_frame = next_frame
                self.video_idx += 1
                continue

            cur_frame_pts = self.cur_frame.pts
            cur_frame_time = float(cur_frame_pts * self.video_time_base)

            if self.extraction_step is not None:
                # We are targeting a specified list of extraction times (because fps or num_frames was specified).
                assert self.next_extraction_time is not None

                if next_frame is None:
                    # cur_frame is the last frame of the video. If it is before the next extraction time, then we
                    # have reached the end of the video.
                    if cur_frame_time < self.next_extraction_time:
                        raise StopIteration
                else:
                    # The extraction time represents the idealized timestamp of the next frame we want to extract.
                    # If next_frame is *closer* to it than cur_frame, then we skip cur_frame.
                    # The following logic handles all three cases:
                    # - next_extraction_time is before cur_frame_time (never skips)
                    # - next_extraction_time is after next_frame_time (always skips)
                    # - next_extraction_time is between cur_frame_time and next_frame_time (depends on which is closer)
                    next_frame_pts = next_frame.pts
                    next_frame_time = float(next_frame_pts * self.video_time_base)
                    if next_frame_time - self.next_extraction_time < self.next_extraction_time - cur_frame_time:
                        self.cur_frame = next_frame
                        self.video_idx += 1
                        continue

            img = self.cur_frame.to_image()
            assert isinstance(img, PIL.Image.Image)
            result: dict[str, Any] = {'frame': img}
            if self.all_frame_attrs:
                attrs = {
                    'index': self.video_idx,
                    'pts': cur_frame_pts,
                    'dts': self.cur_frame.dts,
                    'time': float(cur_frame_pts * self.video_time_base),
                    'is_corrupt': self.cur_frame.is_corrupt,
                    'key_frame': self.cur_frame.key_frame,
                    'pict_type': self.cur_frame.pict_type,
                    'interlaced_frame': self.cur_frame.interlaced_frame,
                }
                result['frame_attrs'] = attrs
            else:
                pos_msec = float(cur_frame_pts * self.video_time_base * 1000 - self.video_start_time)
                result.update({'frame_idx': self.pos, 'pos_msec': pos_msec, 'pos_frame': self.video_idx})

            self.cur_frame = next_frame
            self.video_idx += 1

            self.pos += 1
            if self.extraction_step is not None:
                self.next_extraction_time += self.extraction_step

            return result

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int, **kwargs: Any) -> None:
        assert next(iter(kwargs.values()), None) is not None

        if self.pos == pos:
            # Nothing to do
            return

        self.pos = pos

        seek_time: float
        if 'pos_msec' in kwargs:
            self.video_idx = kwargs['pos_frame']
            seek_time = kwargs['pos_msec'] / 1000.0 + self.video_start_time
        else:
            assert 'frame_attrs' in kwargs
            self.video_idx = kwargs['frame_attrs']['index']
            seek_time = kwargs['frame_attrs']['time']

        assert isinstance(self.video_idx, int)
        assert isinstance(seek_time, float)

        seek_pts = math.floor(seek_time / self.video_time_base)
        self.container.seek(seek_pts, backward=True, stream=self.container.streams.video[0])

        self.cur_frame = self.next_frame()
        while self.cur_frame is not None and float(self.cur_frame.pts * self.video_time_base) < seek_time - 1e-3:
            self.cur_frame = self.next_frame()
        assert self.cur_frame is None or abs(float(self.cur_frame.pts * self.video_time_base) - seek_time) < 1e-3

    @classmethod
    @deprecated('create() is deprecated; use `pixeltable.functions.video.frame_iterator` instead', version='0.5.6')
    def create(cls, **kwargs: Any) -> tuple[type[ComponentIterator], dict[str, Any]]:
        return super()._create(**kwargs)


class VideoSplitter(ComponentIterator):
    """
    Iterator over segments of a video file, which is split into segments. The segments are specified either via a
    fixed duration or a list of split points.

    Args:
        duration: Video segment duration in seconds
        overlap: Overlap between consecutive segments in seconds. Only available for `mode='fast'`.
        min_segment_duration: Drop the last segment if it is smaller than min_segment_duration.
        segment_times: List of timestamps (in seconds) in video where segments should be split. Note that these are not
            segment durations. If all segment times are less than the duration of the video, produces exactly
            `len(segment_times) + 1` segments. An argument of `[]` will produce a single segment containing the
            entire video.
        mode: Segmentation mode:
            - `'fast'`: Quick segmentation using stream copy (splits only at keyframes, approximate durations)
            - `'accurate'`: Precise segmentation with re-encoding (exact durations, slower)
        video_encoder: Video encoder to use. If not specified, uses the default encoder for the current platform.
            Only available for `mode='accurate'`.
        video_encoder_args: Additional arguments to pass to the video encoder. Only available for `mode='accurate'`.
    """

    # Input parameters
    video_path: Path
    segment_duration: float | None
    segment_times: list[float] | None  # [] is valid
    overlap: float
    min_segment_duration: float
    video_encoder: str | None
    video_encoder_args: dict[str, Any] | None

    # Video metadata
    video_time_base: Fraction

    output_iter: Iterator[dict[str, Any]]

    def __init__(
        self,
        video: str,
        *,
        duration: float | None = None,
        overlap: float | None = None,
        min_segment_duration: float | None = None,
        segment_times: list[float] | None = None,
        mode: Literal['fast', 'accurate'] = 'accurate',
        video_encoder: str | None = None,
        video_encoder_args: dict[str, Any] | None = None,
    ):
        Env.get().require_binary('ffmpeg')
        self._check_args(
            duration, segment_times, overlap, min_segment_duration, mode, video_encoder, video_encoder_args
        )
        assert (duration is not None) != (segment_times is not None)
        if duration is not None:
            assert duration > 0.0
            assert duration >= min_segment_duration
            assert overlap is None or overlap < duration

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()

        self.video_path = video_path
        self.segment_duration = duration
        self.overlap = overlap if overlap is not None else 0.0
        self.min_segment_duration = min_segment_duration if min_segment_duration is not None else 0.0
        self.segment_times = segment_times
        self.video_encoder = video_encoder
        self.video_encoder_args = video_encoder_args

        if self.segment_times is not None and len(self.segment_times) == 0:
            self.output_iter = self.complete_video_iter()
        else:
            self.output_iter = self.fast_iter() if mode == 'fast' else self.accurate_iter()

        with av.open(str(video_path)) as container:
            self.video_time_base = container.streams.video[0].time_base

        # TODO: check types of args

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'video': ts.VideoType(nullable=False),
            'duration': ts.FloatType(nullable=True),
            'overlap': ts.FloatType(nullable=True),
            'min_segment_duration': ts.FloatType(nullable=True),
            'segment_times': ts.JsonType(nullable=True),
            'mode': ts.StringType(nullable=False),
            'video_encoder': ts.StringType(nullable=True),
            'video_encoder_args': ts.JsonType(nullable=True),
        }

    @classmethod
    def _check_args(
        cls,
        segment_duration: Any,
        segment_times: Any,
        overlap: Any,
        min_segment_duration: Any,
        mode: Any,
        video_encoder: Any,
        video_encoder_args: Any,
    ) -> None:
        if segment_duration is None and segment_times is None:
            raise excs.Error('Must specify either duration or segment_times')
        if segment_duration is not None and segment_times is not None:
            raise excs.Error('duration and segment_times cannot both be specified')
        if segment_times is not None and overlap is not None:
            raise excs.Error('overlap cannot be specified with segment_times')
        if segment_duration is not None and isinstance(segment_duration, (int, float)):
            if segment_duration <= 0.0:
                raise excs.Error(f'duration must be a positive number: {segment_duration}')
            if (
                min_segment_duration is not None
                and isinstance(min_segment_duration, (int, float))
                and segment_duration < min_segment_duration
            ):
                raise excs.Error(
                    f'duration must be at least min_segment_duration: {segment_duration} < {min_segment_duration}'
                )
            if overlap is not None and isinstance(overlap, (int, float)) and overlap >= segment_duration:
                raise excs.Error(f'overlap must be less than duration: {overlap} >= {segment_duration}')
        if mode == 'accurate' and overlap is not None:
            raise excs.Error("Cannot specify overlap for mode='accurate'")
        if mode == 'fast':
            if video_encoder is not None:
                raise excs.Error("Cannot specify video_encoder for mode='fast'")
            if video_encoder_args is not None:
                raise excs.Error("Cannot specify video_encoder_args for mode='fast'")

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        param_names = ['duration', 'overlap', 'min_segment_duration', 'segment_times']
        params = dict(zip(param_names, args))
        params.update(kwargs)

        segment_duration = params.get('duration')
        segment_times = params.get('segment_times')
        overlap = params.get('overlap')
        min_segment_duration = params.get('min_segment_duration')
        mode = params.get('mode', 'accurate')
        video_encoder = params.get('video_encoder')
        video_encoder_args = params.get('video_encoder_args')
        cls._check_args(
            segment_duration, segment_times, overlap, min_segment_duration, mode, video_encoder, video_encoder_args
        )

        return {
            'segment_start': ts.FloatType(nullable=True),
            'segment_start_pts': ts.IntType(nullable=True),
            'segment_end': ts.FloatType(nullable=True),
            'segment_end_pts': ts.IntType(nullable=True),
            'video_segment': ts.VideoType(nullable=False),
        }, []

    def complete_video_iter(self) -> Iterator[dict[str, Any]]:
        """Returns the entire video as a single segment"""
        assert len(self.segment_times) == 0

        with av.open(str(self.video_path)) as container:
            video_stream = container.streams.video[0]
            start_ts = (
                float(video_stream.start_time * video_stream.time_base)
                if video_stream.start_time is not None and video_stream.time_base is not None
                else 0.0
            )
            end_pts = (
                video_stream.start_time + video_stream.duration
                if video_stream.start_time is not None and video_stream.duration is not None
                else None
            )
            end_ts = (
                float(end_pts * video_stream.time_base)
                if end_pts is not None and video_stream.time_base is not None
                else 0.0
            )
            result = {
                'segment_start': start_ts,
                'segment_start_pts': video_stream.start_time,
                'segment_end': end_ts,
                'segment_end_pts': end_pts,
                'video_segment': str(self.video_path),
            }
            yield result

    def fast_iter(self) -> Iterator[dict[str, Any]]:
        segment_path: str = ''
        assert self.segment_times is None or len(self.segment_times) > 0

        try:
            start_time = 0.0
            start_pts = 0
            segment_idx = 0
            while True:
                target_duration: float | None
                if self.segment_duration is not None:
                    target_duration = self.segment_duration
                elif self.segment_times is not None and segment_idx < len(self.segment_times):
                    target_duration = self.segment_times[segment_idx] - start_time
                else:
                    target_duration = None  # the rest of the video

                segment_path = str(TempStore.create_path(extension='.mp4'))
                cmd = av_utils.ffmpeg_clip_cmd(str(self.video_path), segment_path, start_time, target_duration)
                _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # use the actual duration
                segment_duration = av_utils.get_video_duration(segment_path)
                if segment_duration - self.overlap == 0.0 or segment_duration < self.min_segment_duration:
                    # we're done
                    Path(segment_path).unlink()
                    return

                segment_end = start_time + segment_duration
                segment_end_pts = start_pts + round(segment_duration / self.video_time_base)
                result = {
                    'segment_start': start_time,
                    'segment_start_pts': start_pts,
                    'segment_end': segment_end,
                    'segment_end_pts': segment_end_pts,
                    'video_segment': segment_path,
                }
                yield result

                start_time = segment_end - self.overlap
                start_pts = segment_end_pts - round(self.overlap / self.video_time_base)

                segment_idx += 1
                if self.segment_times is not None and segment_idx > len(self.segment_times):
                    # We've created all segments including the final segment after the last segment_time
                    break

        except subprocess.CalledProcessError as e:
            if segment_path and Path(segment_path).exists():
                Path(segment_path).unlink()
            error_msg = f'ffmpeg failed with return code {e.returncode}'
            if e.stderr:
                error_msg += f': {e.stderr.strip()}'
            raise pxt.Error(error_msg) from e

    def accurate_iter(self) -> Iterator[dict[str, Any]]:
        assert self.segment_times is None or len(self.segment_times) > 0
        base_path = TempStore.create_path(extension='')
        # Use ffmpeg -f segment for accurate segmentation with re-encoding
        output_pattern = f'{base_path}_segment_%04d.mp4'
        cmd = av_utils.ffmpeg_segment_cmd(
            str(self.video_path),
            output_pattern,
            segment_duration=self.segment_duration,
            segment_times=self.segment_times,
            video_encoder=self.video_encoder,
            video_encoder_args=self.video_encoder_args,
        )

        try:
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output_paths = sorted(glob.glob(f'{base_path}_segment_*.mp4'))
            # TODO: is this actually an error?
            # if len(output_paths) == 0:
            #     stderr_output = result.stderr.strip() if result.stderr is not None else ''
            #     raise pxt.Error(
            #         f'ffmpeg failed to create output files for commandline: {" ".join(cmd)}\n{stderr_output}'
            #     )
            start_time = 0.0
            start_pts = 0
            for segment_path in output_paths:
                segment_duration = av_utils.get_video_duration(segment_path)
                if segment_duration < self.min_segment_duration:
                    Path(segment_path).unlink()
                    return

                result = {
                    'segment_start': start_time,
                    'segment_start_pts': start_pts,
                    'segment_end': start_time + segment_duration,
                    'segment_end_pts': start_pts + round(segment_duration / self.video_time_base),
                    'video_segment': segment_path,
                }
                yield result
                start_time += segment_duration
                start_pts += round(segment_duration / self.video_time_base)

        except subprocess.CalledProcessError as e:
            error_msg = f'ffmpeg failed with return code {e.returncode}'
            if e.stderr:
                error_msg += f': {e.stderr.strip()}'
            raise pxt.Error(error_msg) from e

    def __next__(self) -> dict[str, Any]:
        return next(self.output_iter)

    def close(self) -> None:
        pass

    @classmethod
    @deprecated('create() is deprecated; use `pixeltable.functions.video.video_splitter` instead', version='0.5.6')
    def create(cls, **kwargs: Any) -> tuple[type[ComponentIterator], dict[str, Any]]:
        return super()._create(**kwargs)

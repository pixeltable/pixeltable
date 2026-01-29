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
from pixeltable.func.iterator import IteratorCall
import pixeltable.type_system as ts
import pixeltable.utils.av as av_utils
from pixeltable.env import Env
from pixeltable.utils.local_store import TempStore

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class FrameIterator(ComponentIterator):
    @classmethod
    @deprecated('`FrameIterator.create()` is deprecated; use `pixeltable.functions.video.frame_iterator()` instead', version='0.5.6')
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.video import frame_iterator

        return frame_iterator(**kwargs)


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

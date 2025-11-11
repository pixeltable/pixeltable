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
    Iterator over frames of a video. At most one of `fps` or `num_frames` may be specified. If `fps` is specified,
    then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified, then the
    exact number of frames will be extracted. If neither is specified, then all frames will be extracted. The first
    frame of the video will always be extracted, and the remaining frames will be spaced as evenly as possible.

    Args:
        fps: Number of frames to extract per second of video. This may be a fractional value, such as 0.5.
            If omitted or set to 0.0, or if greater than the native framerate of the video,
            then the framerate of the video will be used (all frames will be extracted).
        num_frames: Exact number of frames to extract. The frames will be spaced as evenly as possible. If
            `num_frames` is greater than the number of frames in the video, all frames will be extracted.
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
    all_frame_attrs: bool

    # Video info
    container: av.container.input.InputContainer
    video_framerate: Fraction
    video_time_base: Fraction
    video_frame_count: int
    video_start_time: int

    # List of frame indices to be extracted, or None to extract all frames
    frames_to_extract: list[int] | None

    # Next frame to extract, as an iterator `pos` index. If `frames_to_extract` is None, this is the same as the
    # frame index in the video. Otherwise, the corresponding video index is `frames_to_extract[next_pos]`.
    next_pos: int

    def __init__(
        self, video: str, *, fps: float | None = None, num_frames: int | None = None, all_frame_attrs: bool = False
    ):
        if fps is not None and num_frames is not None:
            raise excs.Error('At most one of `fps` or `num_frames` may be specified')

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.fps = fps
        self.num_frames = num_frames
        self.all_frame_attrs = all_frame_attrs

        self.video_framerate = self.container.streams.video[0].average_rate
        self.video_time_base = self.container.streams.video[0].time_base
        self.video_start_time = self.container.streams.video[0].start_time or 0

        # Determine the number of frames in the video
        self.video_frame_count = self.container.streams.video[0].frames
        if self.video_frame_count == 0:
            # The video codec does not provide a frame count in the standard `frames` field. Try some other methods.
            metadata: dict = self.container.streams.video[0].metadata
            if 'NUMBER_OF_FRAMES' in metadata:
                self.video_frame_count = int(metadata['NUMBER_OF_FRAMES'])
            elif 'DURATION' in metadata:
                # As a last resort, calculate the frame count from the stream duration.
                duration = metadata['DURATION']
                assert isinstance(duration, str)
                seconds = pd.to_timedelta(duration).total_seconds()
                # Usually the duration and framerate are precise enough for this calculation to be accurate, but if
                # we encounter a case where it's off by one due to a rounding error, that's ok; we only use this
                # to determine the positions of the sampled frames when `fps` or `num_frames` is specified.
                self.video_frame_count = round(seconds * self.video_framerate)
            else:
                raise excs.Error(f'Video {video}: failed to get number of frames')

        if num_frames is not None:
            # specific number of frames
            if num_frames > self.video_frame_count:
                # Extract all frames
                self.frames_to_extract = None
            else:
                spacing = float(self.video_frame_count) / float(num_frames)
                self.frames_to_extract = [round(i * spacing) for i in range(num_frames)]
                assert len(self.frames_to_extract) == num_frames
        elif fps is None or fps == 0.0 or fps > float(self.video_framerate):
            # Extract all frames
            self.frames_to_extract = None
        else:
            # Extract frames at the implied frequency
            freq = fps / float(self.video_framerate)
            n = math.ceil(self.video_frame_count * freq)  # number of frames to extract
            self.frames_to_extract = [round(i / freq) for i in range(n)]

        _logger.debug(f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames}')
        self.next_pos = 0

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'video': ts.VideoType(nullable=False),
            'fps': ts.FloatType(nullable=True),
            'num_frames': ts.IntType(nullable=True),
            'all_frame_attrs': ts.BoolType(nullable=False),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        attrs: dict[str, ts.ColumnType]
        if kwargs.get('all_frame_attrs'):
            attrs = {'frame_attrs': ts.JsonType()}
        else:
            attrs = {'frame_idx': ts.IntType(), 'pos_msec': ts.FloatType(), 'pos_frame': ts.IntType()}
        return {**attrs, 'frame': ts.ImageType()}, ['frame']

    def __next__(self) -> dict[str, Any]:
        # Determine the frame index in the video corresponding to the iterator index `next_pos`;
        # the frame at this index is the one we want to extract next
        if self.frames_to_extract is None:
            next_video_idx = self.next_pos  # we're extracting all frames
        elif self.next_pos >= len(self.frames_to_extract):
            raise StopIteration
        else:
            next_video_idx = self.frames_to_extract[self.next_pos]

        # We are searching for the frame at the index implied by `next_pos`. Step through the video until we
        # find it. There are two reasons why it might not be the immediate next frame in the video:
        # (1) `fps` or `num_frames` was specified as an iterator argument; or
        # (2) we just did a seek, and the desired frame is not a keyframe.
        # TODO: In case (1) it will usually be fastest to step through the frames until we find the one we're
        #     looking for. But in some cases it may be faster to do a seek; for example, when `fps` is very
        #     low and there are multiple keyframes in between each frame we want to extract (imagine extracting
        #     10 frames from an hourlong video).
        while True:
            try:
                frame = next(self.container.decode(video=0))
            except EOFError:
                raise StopIteration from None
            # Compute the index of the current frame in the video based on the presentation timestamp (pts);
            # this ensures we have a canonical understanding of frame index, regardless of how we got here
            # (seek or iteration)
            pts = frame.pts - self.video_start_time
            video_idx = round(pts * self.video_time_base * self.video_framerate)
            assert isinstance(video_idx, int)
            if video_idx < next_video_idx:
                # We haven't reached the desired frame yet
                continue

            # Sanity check that we're at the right frame.
            if video_idx != next_video_idx:
                raise excs.Error(f'Frame {next_video_idx} is missing from the video (video file is corrupt)')
            img = frame.to_image()
            assert isinstance(img, PIL.Image.Image)
            pts_msec = float(pts * self.video_time_base * 1000)
            result: dict[str, Any] = {'frame': img}
            if self.all_frame_attrs:
                attrs = {
                    'index': video_idx,
                    'pts': frame.pts,
                    'dts': frame.dts,
                    'time': frame.time,
                    'is_corrupt': frame.is_corrupt,
                    'key_frame': frame.key_frame,
                    'pict_type': frame.pict_type,
                    'interlaced_frame': frame.interlaced_frame,
                }
                result['frame_attrs'] = attrs
            else:
                result.update({'frame_idx': self.next_pos, 'pos_msec': pts_msec, 'pos_frame': video_idx})
            self.next_pos += 1
            return result

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int) -> None:
        if pos == self.next_pos:
            return  # already there

        video_idx = pos if self.frames_to_extract is None else self.frames_to_extract[pos]
        _logger.debug(f'seeking to frame number {video_idx} (at iterator index {pos})')
        # compute the frame position in time_base units
        seek_pos = int(video_idx / self.video_framerate / self.video_time_base + self.video_start_time)
        # This will seek to the nearest keyframe before the desired frame. If the frame being sought is not a keyframe,
        # then the iterator will step forward to the desired frame on the subsequent call to next().
        self.container.seek(seek_pos, backward=True, stream=self.container.streams.video[0])
        self.next_pos = pos


class VideoSplitter(ComponentIterator):
    """
    Iterator over segments of a video file, which is split into fixed-size segments of length `segment_duration`
    seconds.

    Args:
        duration: Video segment duration in seconds
        overlap: Overlap between consecutive segments in seconds. Only available for `mode='fast'`.
        min_segment_duration: Drop the last segment if it is smaller than min_segment_duration.
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
    segment_times: list[float] | None
    overlap: float
    min_segment_duration: float
    video_encoder: str | None
    video_encoder_args: dict[str, Any] | None

    # Video metadata
    video_duration: float
    video_time_base: Fraction
    video_start_time: int

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
        assert (duration is not None) != (segment_times is not None)
        if segment_times is not None:
            assert len(segment_times) > 0
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

        with av.open(str(video_path)) as container:
            video_stream = container.streams.video[0]
            self.video_time_base = video_stream.time_base
            self.video_start_time = video_stream.start_time or 0

        self.output_iter = self.fast_iter() if mode == 'fast' else self.accurate_iter()

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
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        param_names = ['duration', 'overlap', 'min_segment_duration', 'segment_times']
        params = dict(zip(param_names, args))
        params.update(kwargs)

        segment_duration = params.get('duration')
        segment_times = params.get('segment_times')
        overlap = params.get('overlap')
        min_segment_duration = params.get('min_segment_duration')
        mode = params.get('mode', 'fast')

        if segment_duration is None and segment_times is None:
            raise excs.Error('Must specify either duration or segment_times')
        if segment_duration is not None and segment_times is not None:
            raise excs.Error('duration and segment_times cannot both be specified')
        if segment_times is not None:
            if len(segment_times) == 0:
                raise excs.Error('segment_times cannot be empty')
            if overlap is not None:
                raise excs.Error('overlap cannot be specified with segment_times')
        if segment_duration is not None:
            if segment_duration <= 0.0:
                raise excs.Error('duration must be a positive number')
            if min_segment_duration is not None and segment_duration < min_segment_duration:
                raise excs.Error('duration must be at least min_segment_duration')
            if overlap is not None and overlap >= segment_duration:
                raise excs.Error('overlap must be less than duration')
        if mode == 'accurate' and overlap is not None:
            raise excs.Error("Cannot specify overlap for mode='accurate'")
        if mode == 'fast':
            if params.get('video_encoder') is not None:
                raise excs.Error("Cannot specify video_encoder for mode='fast'")
            if params.get('video_encoder_args') is not None:
                raise excs.Error("Cannot specify video_encoder_args for mode='fast'")

        return {
            'segment_start': ts.FloatType(nullable=False),
            'segment_start_pts': ts.IntType(nullable=False),
            'segment_end': ts.FloatType(nullable=False),
            'segment_end_pts': ts.IntType(nullable=False),
            'video_segment': ts.VideoType(nullable=False),
        }, []

    def fast_iter(self) -> Iterator[dict[str, Any]]:
        segment_path: str = ''
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

    def set_pos(self, pos: int) -> None:
        pass

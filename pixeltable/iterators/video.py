import logging
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import av
import pandas as pd
import PIL.Image

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
import pixeltable.utils.av as av_utils
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
            If omitted or set to 0.0, then the native framerate of the video will be used (all frames will be
            extracted). If `fps` is greater than the frame rate of the video, an error will be raised.
        num_frames: Exact number of frames to extract. The frames will be spaced as evenly as possible. If
            `num_frames` is greater than the number of frames in the video, all frames will be extracted.
        all_frame_attrs:
            If True, outputs a `pxt.Json` column `frame_attrs` with the following `pyav`-provided attributes
            (for more information, see `pyav`'s documentation on
            [VideoFrame](https://pyav.org/docs/develop/api/video.html#module-av.video.frame) and
            [Frame](https://pyav.org/docs/develop/api/frame.html)):

            * `index` (`int`)
            * `pts` (`Optional[int]`)
            * `dts` (`Optional[int]`)
            * `time` (`Optional[float]`)
            * `is_corrupt` (`bool`)
            * `key_frame` (`bool`)
            * `pict_type` (`int`)
            * `interlaced_frame` (`bool`)

            If False, only outputs frame attributes `frame_idx`, `pos_msec`, and `pos_frame` as separate columns.
    """

    # Input parameters
    video_path: Path
    fps: Optional[float]
    num_frames: Optional[int]
    all_frame_attrs: bool

    # Video info
    container: av.container.input.InputContainer
    video_framerate: Fraction
    video_time_base: Fraction
    video_frame_count: int
    video_start_time: int

    # List of frame indices to be extracted, or None to extract all frames
    frames_to_extract: Optional[list[int]]

    # Next frame to extract, as an iterator `pos` index. If `frames_to_extract` is None, this is the same as the
    # frame index in the video. Otherwise, the corresponding video index is `frames_to_extract[next_pos]`.
    next_pos: int

    def __init__(
        self,
        video: str,
        *,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        all_frame_attrs: bool = False,
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
        elif fps is None or fps == 0.0:
            # Extract all frames
            self.frames_to_extract = None
        elif fps > float(self.video_framerate):
            raise excs.Error(
                f'Video {video}: requested fps ({fps}) exceeds that of the video ({float(self.video_framerate)})'
            )
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
        segment_duration: Video segment duration in seconds
        overlap: Overlap between consecutive segments in seconds.
        min_segment_duration: Drop the last segment if it is smaller than min_segment_duration
    """

    # Input parameters
    video_path: Path
    segment_duration: float
    overlap: float
    min_segment_duration: float

    # Video metadata
    video_duration: float
    video_time_base: Fraction
    video_start_time: int

    # position tracking
    next_segment_start: float
    next_segment_start_pts: int

    def __init__(self, video: str, segment_duration: float, *, overlap: float = 0.0, min_segment_duration: float = 0.0):
        assert segment_duration > 0.0
        assert segment_duration >= min_segment_duration
        assert overlap < segment_duration

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()

        if not shutil.which('ffmpeg'):
            raise pxt.Error('ffmpeg is not installed or not in PATH. Please install ffmpeg to use VideoSplitter.')

        self.video_path = video_path
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.min_segment_duration = min_segment_duration

        with av.open(str(video_path)) as container:
            video_stream = container.streams.video[0]
            self.video_time_base = video_stream.time_base
            self.video_start_time = video_stream.start_time or 0

        self.next_segment_start = float(self.video_start_time * self.video_time_base)
        self.next_segment_start_pts = self.video_start_time

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'video': ts.VideoType(nullable=False),
            'segment_duration': ts.FloatType(nullable=False),
            'overlap': ts.FloatType(nullable=True),
            'min_segment_duration': ts.FloatType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        param_names = ['segment_duration', 'overlap', 'min_segment_duration']
        params = dict(zip(param_names, args))
        params.update(kwargs)

        segment_duration = params['segment_duration']
        min_segment_duration = params.get('min_segment_duration', 0.0)
        overlap = params.get('overlap', 0.0)

        if segment_duration <= 0.0:
            raise excs.Error('segment_duration must be a positive number')
        if segment_duration < min_segment_duration:
            raise excs.Error('segment_duration must be at least min_segment_duration')
        if overlap >= segment_duration:
            raise excs.Error('overlap must be less than segment_duration')

        return {
            'segment_start': ts.FloatType(nullable=False),
            'segment_start_pts': ts.IntType(nullable=False),
            'segment_end': ts.FloatType(nullable=False),
            'segment_end_pts': ts.IntType(nullable=False),
            'video_segment': ts.VideoType(nullable=False),
        }, []

    def __next__(self) -> dict[str, Any]:
        segment_path = str(TempStore.create_path(extension='.mp4'))
        try:
            cmd = av_utils.ffmpeg_clip_cmd(
                str(self.video_path), segment_path, self.next_segment_start, self.segment_duration
            )
            _ = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # use the actual duration
            segment_duration = av_utils.get_video_duration(segment_path)
            if segment_duration - self.overlap == 0.0:
                # we're done
                Path(segment_path).unlink()
                raise StopIteration

            if segment_duration < self.min_segment_duration:
                Path(segment_path).unlink()
                raise StopIteration

            segment_end = self.next_segment_start + segment_duration
            segment_end_pts = self.next_segment_start_pts + round(segment_duration / self.video_time_base)

            result = {
                'segment_start': self.next_segment_start,
                'segment_start_pts': self.next_segment_start_pts,
                'segment_end': segment_end,
                'segment_end_pts': segment_end_pts,
                'video_segment': segment_path,
            }
            self.next_segment_start = segment_end - self.overlap
            self.next_segment_start_pts = segment_end_pts - round(self.overlap / self.video_time_base)

            return result

        except subprocess.CalledProcessError as e:
            if Path(segment_path).exists():
                Path(segment_path).unlink()
            error_msg = f'ffmpeg failed with return code {e.returncode}'
            if e.stderr:
                error_msg += f': {e.stderr.strip()}'
            raise pxt.Error(error_msg) from e

    def close(self) -> None:
        pass

    def set_pos(self, pos: int) -> None:
        pass

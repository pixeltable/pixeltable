import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional, Sequence

import av  # type: ignore[import-untyped]
import pandas as pd
import PIL.Image

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class FrameIterator(ComponentIterator):
    """
    Iterator over frames of a video. At most one of `fps` or `num_frames` may be specified. If `fps` is specified,
    then frames will be extracted at the specified rate (frames per second). If `num_frames` is specified, then the
    exact number of frames will be extracted. If neither is specified, then all frames will be extracted. The first
    frame of the video will always be extracted, and the remaining frames will be spaced as evenly as possible.

        Args:
            video: URL or path of the video to use for frame extraction.
            fps: Number of frames to extract per second of video. This may be a fractional value, such as 0.5.
                If omitted or set to 0.0, then the native framerate of the video will be used (all frames will be
                extracted). If `fps` is greater than the frame rate of the video, an error will be raised.
            num_frames: Exact number of frames to extract. The frames will be spaced as evenly as possible. If
                `num_frames` is greater than the number of frames in the video, all frames will be extracted.
    """

    # Input parameters
    video_path: Path
    fps: Optional[float]
    num_frames: Optional[int]

    # Video info
    container: av.container.input.InputContainer
    video_framerate: Fraction
    video_time_base: Fraction
    video_frame_count: int
    video_start_time: int

    # Frame extraction info
    frames_to_extract: Sequence[int]
    frames_set: set[int]

    # State
    next_frame_idx: int
    next_pos_frame: int  # sanity check

    def __init__(self, video: str, *, fps: Optional[float] = None, num_frames: Optional[int] = None):
        if fps is not None and num_frames is not None:
            raise excs.Error('At most one of `fps` or `num_frames` may be specified')

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.container = av.open(str(video_path))
        self.fps = fps
        self.num_frames = num_frames

        self.video_framerate = self.container.streams.video[0].average_rate
        self.video_time_base = self.container.streams.video[0].time_base
        self.video_start_time = self.container.streams.video[0].start_time or 0

        self.video_frame_count = self.container.streams.video[0].frames
        if self.video_frame_count == 0:
            # The video codec does not provide a frame count in the `frames` field. Try some other methods.
            metadata: dict = self.container.streams.video[0].metadata
            if 'NUMBER_OF_FRAMES' in metadata:
                self.video_frame_count = int(metadata['NUMBER_OF_FRAMES'])
            elif 'DURATION' in metadata:
                duration = metadata['DURATION']
                assert isinstance(duration, str)
                seconds = pd.to_timedelta(duration).total_seconds()
                # Usually the duration and framerate are precise enough for this calculation to be accurate, but if
                # we encounter a case where it's off by one due to a rounding error, that's ok; we only use this
                # to determine the positions of the sampled frames when `fps` or `num_frames` is specified.
                self.video_frame_count = round(seconds * self.video_framerate)
            else:
                raise excs.Error(f'Video {video}: failed to get number of frames')

        if fps is not None and fps > float(self.video_framerate):
            raise excs.Error(f'Video {video}: requested fps ({fps}) exceeds that of the video ({float(self.video_framerate)})')

        if num_frames is not None:
            # specific number of frames
            if num_frames > self.video_frame_count:
                # Extract all frames
                self.frames_to_extract = range(self.video_frame_count)
            else:
                spacing = float(self.video_frame_count) / float(num_frames)
                self.frames_to_extract = list(round(i * spacing) for i in range(num_frames))
                assert len(self.frames_to_extract) == num_frames
        else:
            if fps is None or fps == 0.0:
                # Extract all frames
                self.frames_to_extract = range(self.video_frame_count)
            else:
                # Extract frames at the implied frequency
                freq = fps / float(self.video_framerate)
                n = math.ceil(self.video_frame_count * freq)  # number of frames to extract
                self.frames_to_extract = list(round(i / freq) for i in range(n))

        # We need the list of frames as both a list (for set_pos) and a set (for fast lookups when
        # there are lots of frames)
        self.frames_set = set(self.frames_to_extract)
        _logger.debug(f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames}')
        self.next_frame_idx = 0
        self.next_pos_frame = 0

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'video': ts.VideoType(nullable=False),
            'fps': ts.FloatType(nullable=True),
            'num_frames': ts.IntType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'frame_idx': ts.IntType(),
            'pos_frame': ts.IntType(),
            'pos_msec': ts.FloatType(),
            'frame': ts.ImageType(),
        }, ['frame']

    def __next__(self) -> dict[str, Any]:
        while True:
            try:
                frame = next(self.container.decode(video=0))
            except EOFError:
                raise StopIteration
            pts = frame.pts - self.video_start_time
            pos_msec = float(pts * self.video_time_base * 1000)
            pos_frame = round(pts * self.video_time_base * self.video_framerate)
            assert isinstance(pos_frame, int)
            assert pos_frame <= self.next_pos_frame, f'{pos_frame} > {self.next_pos_frame}, {pts}'
            if pos_frame < self.next_pos_frame:
                # This can happen after a seek, because the frame we seek to is always a keyframe, and
                # `self.next_pos_frame` is not necessarily a keyframe
                continue
            img = frame.to_image()
            assert isinstance(img, PIL.Image.Image)
            self.next_pos_frame += 1
            if pos_frame in self.frames_set:
                result = {
                    'frame_idx': self.next_frame_idx,
                    'pos_msec': pos_msec,
                    'pos_frame': pos_frame,
                    'frame': img,
                }
                self.next_frame_idx += 1
                return result

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int) -> None:
        """Seek to frame idx"""
        if pos == self.next_frame_idx:
            return
        pos_frame = self.frames_to_extract[pos]
        _logger.debug(f'seeking to frame number {pos_frame} (at index {pos})')
        # compute the frame position in time_base units
        seek_pos = int(pos_frame / self.video_framerate / self.video_time_base + self.video_start_time)
        # This will seek to the nearest keyframe before the desired frame. If the frame being sought is not a keyframe,
        # then the iterator will step forward to the desired frame on the succeeding call to next().
        self.container.seek(seek_pos, backward=True, stream=self.container.streams.video[0])
        self.next_frame_idx = pos
        self.next_pos_frame = pos_frame

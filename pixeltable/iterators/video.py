import logging
import math
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import av
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

    # List of frame indices to be extracted, or None to extract all frames
    frames_to_extract: Optional[list[int]]

    # Next frame to extract, as an iterator `pos` index. If `frames_to_extract` is None, this is the same as the
    # frame index in the video. Otherwise, the corresponding video index is `frames_to_extract[next_pos]`.
    next_pos: int

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
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'frame_idx': ts.IntType(),
            'pos_msec': ts.FloatType(),
            'pos_frame': ts.IntType(),
            'frame': ts.ImageType(),
        }, ['frame']

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
            pos_msec = float(pts * self.video_time_base * 1000)
            result = {'frame_idx': self.next_pos, 'pos_msec': pos_msec, 'pos_frame': video_idx, 'frame': img}
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

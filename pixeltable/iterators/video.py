import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import PIL.Image

from pixeltable.exceptions import Error
from pixeltable.type_system import ColumnType, FloatType, ImageType, IntType, VideoType

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
    def __init__(self, video: str, *, fps: Optional[float] = None, num_frames: Optional[int] = None):
        if fps is not None and num_frames is not None:
            raise Error('At most one of `fps` or `num_frames` may be specified')

        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.video_reader = cv2.VideoCapture(str(video_path))
        self.fps = fps
        self.num_frames = num_frames
        if not self.video_reader.isOpened():
            raise Error(f'Failed to open video: {video}')

        video_fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        if fps is not None and fps > video_fps:
            raise Error(f'Video {video}: requested fps ({fps}) exceeds that of the video ({video_fps})')
        num_video_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_video_frames == 0:
            raise Error(f'Video {video}: failed to get number of frames')

        if num_frames is not None:
            # specific number of frames
            if num_frames > num_video_frames:
                # Extract all frames
                self.frames_to_extract = range(num_video_frames)
            else:
                spacing = float(num_video_frames) / float(num_frames)
                self.frames_to_extract = list(round(i * spacing) for i in range(num_frames))
                assert len(self.frames_to_extract) == num_frames
        else:
            if fps is None or fps == 0.0:
                # Extract all frames
                self.frames_to_extract = range(num_video_frames)
            else:
                # Extract frames at the implied frequency
                freq = fps / video_fps
                n = math.ceil(num_video_frames * freq)  # number of frames to extract
                self.frames_to_extract = list(round(i / freq) for i in range(n))

        # We need the list of frames as both a list (for set_pos) and a set (for fast lookups when
        # there are lots of frames)
        self.frames_set = set(self.frames_to_extract)
        _logger.debug(f'FrameIterator: path={self.video_path} fps={self.fps} num_frames={self.num_frames}')
        self.next_frame_idx = 0

    @classmethod
    def input_schema(cls) -> dict[str, ColumnType]:
        return {
            'video': VideoType(nullable=False),
            'fps': FloatType(nullable=True),
            'num_frames': IntType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ColumnType], list[str]]:
        return {
            'frame_idx': IntType(),
            'pos_msec': FloatType(),
            'pos_frame': FloatType(),
            'frame': ImageType(),
        }, ['frame']

    def __next__(self) -> dict[str, Any]:
        # jumping to the target frame here with video_reader.set() is far slower than just
        # skipping the unwanted frames
        while True:
            pos_msec = self.video_reader.get(cv2.CAP_PROP_POS_MSEC)
            pos_frame = self.video_reader.get(cv2.CAP_PROP_POS_FRAMES)
            status, img = self.video_reader.read()
            if not status:
                _logger.debug(f'releasing video reader for {self.video_path}')
                self.video_reader.release()
                self.video_reader = None
                raise StopIteration
            if pos_frame in self.frames_set:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = {
                    'frame_idx': self.next_frame_idx,
                    'pos_msec': pos_msec,
                    'pos_frame': pos_frame,
                    'frame': PIL.Image.fromarray(img),
                }
                self.next_frame_idx += 1
                return result

    def close(self) -> None:
        if self.video_reader is not None:
            self.video_reader.release()
            self.video_reader = None

    def set_pos(self, pos: int) -> None:
        """Seek to frame idx"""
        if pos == self.next_frame_idx:
            return
        _logger.debug(f'seeking to frame {pos}')
        self.video_reader.set(cv2.CAP_PROP_POS_FRAMES, self.frames_to_extract[pos])
        self.next_frame_idx = pos

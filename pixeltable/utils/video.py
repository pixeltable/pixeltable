from __future__ import annotations
import math
from typing import Optional, Tuple
from collections.abc import Iterator
from pathlib import Path
import logging
import cv2

import PIL

from pixeltable.exceptions import Error
from pixeltable.env import Env


_logger = logging.getLogger('pixeltable')

class FrameIterator:
    """
    Iterator over the frames of a video.
    """

    def __init__(self, video_path_str: str, fps: int = 0):
        video_path = Path(video_path_str)
        if not video_path.exists():
            raise Error(f'File not found: {video_path_str}')
        if not video_path.is_file():
            raise Error(f'Not a file: {video_path_str}')
        self.video_path = video_path
        self.fps = fps
        self.video_reader = cv2.VideoCapture(str(video_path))
        if not self.video_reader.isOpened():
            raise Error(f'Failed to open video: {video_path_str}')
        video_fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        if fps > video_fps:
            raise Error(f'Video {video_path_str}: requested fps ({fps}) exceeds that of the video ({video_fps})')
        self.frame_freq = int(video_fps / fps) if fps > 0 else 1
        num_video_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_video_frames == 0:
            raise Error(f'Video {video_path_str}: failed to get number of frames')
        # ceil: round up to ensure we count frame 0
        self.num_frames = math.ceil(num_video_frames / self.frame_freq) if fps > 0 else num_video_frames
        _logger.debug(f'FrameIterator: path={self.video_path} fps={self.fps}')

        self.next_frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[int, PIL.Image.Image]]:
        return self

    def __next__(self) -> Tuple[int, PIL.Image.Image]:
        """Returns (frame idx, image).
        """
        while True:
            status, img = self.video_reader.read()
            if not status:
                _logger.debug(f'releasing video reader for {self.video_path}')
                self.video_reader.release()
                self.video_reader = None
                raise StopIteration
            # -1: CAP_PROP_POS_FRAMES points to the next frame
            if (self.video_reader.get(cv2.CAP_PROP_POS_FRAMES) - 1) % self.frame_freq == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = (self.next_frame_idx, PIL.Image.fromarray(img))
                self.next_frame_idx += 1
                return result

    def seek(self, frame_idx: int) -> None:
        """Fast-forward to frame idx
        """
        assert frame_idx >= self.next_frame_idx  # can't seek backwards
        if frame_idx == self.next_frame_idx:
            return
        _logger.debug(f'seeking to frame {frame_idx}')
        self.video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_idx * self.frame_freq)
        self.next_frame_idx = frame_idx

    def __len__(self) -> int:
        return self.num_frames

    def __enter__(self) -> FrameIterator:
        _logger.debug(f'__enter__ {self.video_path}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _logger.debug(f'__exit__ {self.video_path}')
        self.close()

    def close(self) -> None:
        if self.video_reader is not None:
            self.video_reader.release()
            self.video_reader = None


class FrameExtractor:
    """
    Implements the extract_frame window function.
    """
    def __init__(self, video_path_str: str, fps: int = 0):
        self.frames = FrameIterator(video_path_str, fps=fps)
        self.current_frame_path: Optional[str] = None

    @classmethod
    def make_aggregator(cls, video_path_str: str, fps: int = 0) -> FrameExtractor:
        return cls(video_path_str, fps=fps)

    def update(self, frame_idx: int) -> None:
        self.frames.seek(frame_idx)
        _, self.current_frame_path = next(self.frames)

    def value(self) -> PIL.Image.Image:
        return PIL.Image.open(self.current_frame_path)


# extract_frame = Function.make_library_aggregate_function(
#     ImageType(), [VideoType(), IntType()],  # params: video, frame idx
#     module_name = 'pixeltable.utils.video',
#     init_symbol = 'FrameExtractor.make_aggregator',
#     update_symbol = 'FrameExtractor.update',
#     value_symbol = 'FrameExtractor.value',
#     requires_order_by=True, allows_std_agg=False, allows_window=True)
# don't register this function, it's not meant for users

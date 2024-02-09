from typing import Dict, Any, List, Tuple
from pathlib import Path
import math
import logging

import cv2
import PIL.Image

from .base import ComponentIterator

from pixeltable.type_system import ColumnType, VideoType, ImageType, IntType, FloatType
from pixeltable.exceptions import Error


_logger = logging.getLogger('pixeltable')

class FrameIterator(ComponentIterator):
    def __init__(self, video: str, fps: float = 0.0):
        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.fps = fps
        self.video_reader = cv2.VideoCapture(str(video_path))
        if not self.video_reader.isOpened():
            raise Error(f'Failed to open video: {video}')
        video_fps = int(self.video_reader.get(cv2.CAP_PROP_FPS))
        if fps > video_fps:
            raise Error(f'Video {video}: requested fps ({fps}) exceeds that of the video ({video_fps})')
        self.frame_freq = int(video_fps / fps) if fps > 0 else 1
        num_video_frames = int(self.video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        if num_video_frames == 0:
            raise Error(f'Video {video}: failed to get number of frames')
        # ceil: round up to ensure we count frame 0
        self.num_frames = math.ceil(num_video_frames / self.frame_freq) if fps > 0 else num_video_frames
        _logger.debug(f'FrameIterator: path={self.video_path} fps={self.fps}')

        self.next_frame_idx = 0

    @classmethod
    def input_schema(cls) -> Dict[str, ColumnType]:
        return {
            'video': VideoType(nullable=False),
            'fps': FloatType()
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> Tuple[Dict[str, ColumnType], List[str]]:
        return {
            'frame_idx': IntType(),
            'pos_msec': FloatType(),
            'pos_frame': FloatType(),
            'frame': ImageType(),
        }, ['frame']

    def __next__(self) -> Dict[str, Any]:
        while True:
            pos_msec = self.video_reader.get(cv2.CAP_PROP_POS_MSEC)
            pos_frame = self.video_reader.get(cv2.CAP_PROP_POS_FRAMES)
            status, img = self.video_reader.read()
            if not status:
                _logger.debug(f'releasing video reader for {self.video_path}')
                self.video_reader.release()
                self.video_reader = None
                raise StopIteration
            if pos_frame % self.frame_freq == 0:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = {
                    'frame_idx': self.next_frame_idx,
                    'pos_msec': pos_msec,
                    'pos_frame': pos_frame,
                    'frame': PIL.Image.fromarray(img),
                }
                self.next_frame_idx += 1
                # frame_freq > 1: jumping to the target frame here with video_reader.set() is far slower than just
                # skipping the unwanted frames
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
        self.video_reader.set(cv2.CAP_PROP_POS_FRAMES, pos * self.frame_freq)
        self.next_frame_idx = pos

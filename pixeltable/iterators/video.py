from typing import Dict, Any, List, Tuple, Iterator
from pathlib import Path
import logging

import av

import math
from .base import ComponentIterator

from pixeltable.type_system import ColumnType, VideoType, ImageType, IntType, FloatType, BoolType
from pixeltable.exceptions import Error


_logger = logging.getLogger('pixeltable')

class FrameIterator(ComponentIterator):
    def __init__(self, video: str, fps: float = 0.0):
        video_path = Path(video)
        assert video_path.exists() and video_path.is_file()
        self.video_path = video_path
        self.requested_fps = fps

        self.seek_to: int = 0
        self.iterator = self._make_iterator(self.requested_fps)

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
            'pts': IntType(),
            'key_frame': BoolType(),
            'frame': ImageType(),
        }, ['frame']

    def _make_iterator(self, fps: float) -> Iterator[Dict[str, Any]]:
        next_target_pts = self.seek_to
        output_frame_idx = 0

        with av.open(str(self.video_path), 'r') as video_container:
            video_stream = next(s for s in video_container.streams.video)
            video_fps = float(video_stream.average_rate)
            if fps > video_fps:
                raise Error(f'Video {self.video_path}: requested fps ({fps}) exceeds that of the video ({video_fps})')
            seconds_between_frames = 1. / fps if fps > 0 else 0.0
            _logger.debug(f'FrameIterator: path={self.video_path} fps={self.requested_fps}')
            if self.seek_to > 0:
                video_container.seek(offset=self.seek_to, backward=True, any_frame=False, stream=video_stream)

            for packet in video_container.demux(video_stream):
                for frame in packet.decode():
                    if frame.pts >= next_target_pts:
                        result = {
                            'frame_idx': output_frame_idx,
                            'pos_msec': frame.time * 1000,
                            'pts': frame.pts,
                            'key_frame': bool(frame.key_frame),
                            'frame': frame.to_image()
                        }
                        yield result
                        output_frame_idx += 1
                        next_target_time = frame.time + seconds_between_frames
                        next_target_pts = math.ceil(next_target_time / video_stream.time_base)

    def __next__(self) -> Dict[str, Any]:
        """ return the next dictionary from the video
        """
        return next(self.iterator)

    def close(self) -> None:
        self.video_container.close()

    def set_pos(self, pos: int) -> None:
        """ Note pos is a presentation time-stamp aka pts, not a frame index, as exact frame index access is not supported """
        self.seek_to = pos
        self.iterator = self._make_iterator(self.requested_fps)

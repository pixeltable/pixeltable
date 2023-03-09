from typing import List, Dict, Optional, Tuple
import ffmpeg
import glob
import os
from pathlib import Path
from collections.abc import Iterator
from pathlib import Path

import PIL

from pixeltable.exceptions import RuntimeError
from pixeltable.env import Env
from pixeltable.function import Function, FunctionRegistry
from pixeltable.type_system import IntType, ImageType, VideoType


def extract_frames(
        video_path_str: str, output_path_prefix: str, fps: int = 0, ffmpeg_filter: Optional[Dict[str, str]] = None
) -> List[str]:
    """
    Extract frames at given fps as jpg files (fps == 0: all frames).
    Returns list of frame file paths.
    """
    video_path = Path(video_path_str)
    if not video_path.exists():
        raise RuntimeError(f'File not found: {video_path_str}')
    if not video_path.is_file():
        raise RuntimeError(f'Not a file: {video_path_str}')
    output_path_str = f'{output_path_prefix}_%07d.jpg'
    s = ffmpeg.input(video_path)
    if fps > 0:
        s = s.filter('fps', fps)
    if ffmpeg_filter is not None:
        for key, val in ffmpeg_filter.items():
            s = s.filter(key, val)
    # vsync=0: required to apply filter, otherwise ffmpeg pads the output with duplicate frames
    s = s.output(output_path_str, vsync=0, loglevel='quiet')
    #print(s.get_args())
    try:
        s.run()
    except ffmpeg.Error:
        raise RuntimeError(f'ffmpeg exception')

    # collect generated files
    frame_paths = glob.glob(f'{output_path_prefix}_*.jpg')
    frame_paths.sort()
    return frame_paths

def get_frame_count(video_path_str: str) -> int:
    p = ffmpeg.probe(video_path_str)
    video_stream_info = next((stream for stream in p['streams'] if stream['codec_type'] == 'video'), None)
    return video_stream_info.nb_frames


class FrameIterator:
    """
    Iterator for the frames of a video. Files returned by next() are deleted in the following next() call.
    """
    next_iterator_id = 0  # used to generate unique output paths

    def __init__(self, video_path_str: str, fps: int = 0, ffmpeg_filter: Optional[Dict[str, str]] = None):
        # extract all frames into tmp_frames dir
        video_path = Path(video_path_str)
        if not video_path.exists():
            raise RuntimeError(f'File not found: {video_path_str}')
        if not video_path.is_file():
            raise RuntimeError(f'Not a file: {video_path_str}')
        output_path = Env.get().tmp_frames_dir / f'{str(self.next_iterator_id)}_%07d.jpg'
        s = ffmpeg.input(video_path)
        if fps > 0:
            s = s.filter('fps', fps)
        if ffmpeg_filter is not None:
            for key, val in ffmpeg_filter.items():
                s = s.filter(key, val)
        # vsync=0: required to apply filter, otherwise ffmpeg pads the output with duplicate frames
        s = s.output(str(output_path), vsync=0, loglevel='quiet')
        # _ = s.get_args()
        try:
            s.run()
        except ffmpeg.Error:
            raise RuntimeError(f'ffmpeg exception')
        pattern = Env.get().tmp_frames_dir / f'{str(self.next_iterator_id)}_*.jpg'
        self.frame_files = glob.glob(str(pattern))
        self.frame_files.sort()  # make sure we iterate through these in frame number order
        self.next_frame_idx = 0

        self.__class__.next_iterator_id += 1

    def __iter__(self) -> Iterator[Tuple[int, Path]]:
        return self

    def __next__(self) -> Tuple[int, Path]:
        """
        Returns (frame idx, path to img file).
        """
        prev_frame_idx = self.next_frame_idx - 1
        if prev_frame_idx >= 0:
            # try to delete the file
            try:
                os.remove(self.frame_files[prev_frame_idx])
            except FileNotFoundError:
                # nothing to worry about, someone else grabbed it
                pass
        if self.next_frame_idx == len(self.frame_files):
            raise StopIteration
        result = (self.next_frame_idx, Path(self.frame_files[self.next_frame_idx]))
        self.next_frame_idx += 1
        return result

    def seek(self, frame_idx: int) -> None:
        """
        Fast-forward to frame idx
        """
        assert frame_idx >= self.next_frame_idx
        while frame_idx < self.next_frame_idx:
            _ = self.__next__()

    def num_frames(self) -> int:
        return len(self.frame_files)


class FrameExtractor:
    """
    Implements the extract_frame window function.
    """
    def __init__(self, video_path_str: str, fps: int = 0, ffmpeg_filter: Optional[Dict[str, str]] = None):
        self.frames = FrameIterator(video_path_str, fps=fps, ffmpeg_filter=ffmpeg_filter)
        self.current_frame_path: Optional[str] = None

    @classmethod
    def make_aggregator(
            cls, video_path_str: str, fps: int = 0, ffmpeg_filter: Optional[Dict[str, str]] = None
    ) -> 'FrameExtractor':
        return cls(video_path_str, fps=fps, ffmpeg_filter=ffmpeg_filter)

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

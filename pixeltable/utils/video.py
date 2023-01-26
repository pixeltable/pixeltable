from typing import List
import ffmpeg
import glob
from pathlib import Path

from pixeltable.exceptions import OperationalError


def extract_frames(video_path_str: str, output_path_prefix: str, fps: int = 0) -> List[str]:
    """
    Extract frames at given fps as jpg files (fps == 0: all frames).
    Returns list of frame file paths.
    """
    video_path = Path(video_path_str)
    if not video_path.exists():
        raise OperationalError(f'File not found: {video_path_str}')
    if not video_path.is_file():
        raise OperationalError(f'Not a file: {video_path_str}')
    output_path_str = f'{output_path_prefix}_%07d.jpg'
    s = ffmpeg.input(video_path)
    if fps > 0:
        s = s.filter('fps', fps)
    s = s.output(output_path_str, loglevel='quiet')
    try:
        s.run()
    except ffmpeg.Error:
        raise OperationalError(f'ffmpeg exception')

    # collect generated files
    frame_paths = glob.glob(f'{output_path_prefix}_*.jpg')
    frame_paths.sort()
    return frame_paths

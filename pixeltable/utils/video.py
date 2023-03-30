from typing import List, Dict, Optional, Tuple
import glob
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4
import json

import docker
import ffmpeg
import PIL

from pixeltable.exceptions import RuntimeError
from pixeltable.env import Env


def num_tmp_frames() -> int:
    files = glob.glob(str(Env.get().tmp_frames_dir / '*.jpg'))
    return len(files)

class FrameIterator:
    """
    Iterator for the frames of a video. Files returned by next() are deleted in the following next() call.
    """
    def __init__(self, video_path_str: str, fps: int = 0, ffmpeg_filter: Optional[Dict[str, str]] = None):
        # extract all frames into tmp_frames dir
        video_path = Path(video_path_str)
        if not video_path.exists():
            raise RuntimeError(f'File not found: {video_path_str}')
        if not video_path.is_file():
            raise RuntimeError(f'Not a file: {video_path_str}')
        self.video_path = video_path
        self.fps = fps
        self.ffmpeg_filter = ffmpeg_filter
        self.frame_files: List[str] = []
        self.next_frame_idx = -1  # -1: extraction hasn't happened yet
        self.id = uuid4().hex[:16]

        # get estimate of # of frames
        if ffmpeg_filter is None:
            cl = docker.from_env()
            command = (
                f'-v error -select_streams v:0 -show_entries stream=nb_frames,duration -print_format json '
                f'/input/{self.video_path.name}'
            )
            output = cl.containers.run(
                'sjourdan/ffprobe:latest', command, detach=False, remove=True,
                volumes={str(video_path.parent): {'bind': '/input', 'mode': 'ro'}},
            )
            info = json.loads(output)
            if fps == 0:
                self.est_num_frames = int(info['streams'][0]['nb_frames'])
            else:
                self.est_num_frames = int(fps * float(info['streams'][0]['duration']))
        else:
            # we won't know until we run the extraction
            self.est_num_frames = None

    def _extract_frames(self) -> None:
        assert self.next_frame_idx == -1

        # use ffmpeg-python to construct the command
        s = ffmpeg.input(str(Path('/input') / self.video_path.name))
        if self.fps > 0:
            s = s.filter('fps', self.fps)
        if self.ffmpeg_filter is not None:
            for key, val in self.ffmpeg_filter.items():
                s = s.filter(key, val)
        # vsync=0: required to apply filter, otherwise ffmpeg pads the output with duplicate frames
        s = s.output(str(Path('/output') / f'{self.id}_%07d.jpg'), vsync=0, loglevel='quiet')
        command = ' '.join([f"'{arg}'" for arg in s.get_args()])  # quote everything to deal with spaces

        cl = docker.from_env()
        _ = cl.containers.run(
            'jrottenberg/ffmpeg:4.1-alpine',
            command,
            detach=False,
            remove=True,
            volumes={
                self.video_path.parent: {'bind': '/input', 'mode': 'rw'},
                str(Env.get().tmp_frames_dir): {'bind': '/output', 'mode': 'rw'},
            },
            user=os.getuid(),
        )

        pattern = Env.get().tmp_frames_dir / f'{self.id}_*.jpg'
        self.frame_files = glob.glob(str(pattern))
        self.frame_files.sort()  # make sure we iterate through these in frame number order
        self.next_frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[int, Path]]:
        return self

    def __next__(self) -> Tuple[int, Path]:
        """
        Returns (frame idx, path to img file).
        """
        if self.next_frame_idx == -1:
            self._extract_frames()
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
        if self.next_frame_idx == -1:
            self._extract_frames()
        while frame_idx < self.next_frame_idx:
            _ = self.__next__()

    def __enter__(self) -> 'FrameIterator':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self.close()

    def close(self) -> None:
        if self.next_frame_idx == -1:
            # nothing to do
            return
        while True:
            try:
                _ = self.__next__()
            except StopIteration:
                return


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

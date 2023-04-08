from __future__ import annotations
import re
import time
from typing import List, Dict, Optional, Tuple
import glob
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4
import json
import threading
import queue
import logging

import docker
import ffmpeg
import PIL
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler, FileSystemEvent

from pixeltable.exceptions import RuntimeError
from pixeltable.env import Env


_logger = logging.getLogger('pixeltable')

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
        self.id = uuid4().hex[:16]
        self.idx_re = re.compile(fr'{self.id}_(\d+)\.jpg')  # pattern to extract frame idx from filename
        self.num_frames: Optional[int] = None  # the known number of frames
        _logger.debug(f'FrameIterator: id={self.id} path={self.video_path} fps={self.fps} filter={self.ffmpeg_filter}')

        # get estimate of # of frames
        if ffmpeg_filter is None:
            cl = docker.from_env()
            command = (
                f'-v error -select_streams v:0 -show_entries stream=nb_frames,duration -print_format json '
                f'/input/{self.video_path.name}'
            )
            _logger.debug(f'running ffprobe: {command}')
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
            self.est_num_frames: Optional[int] = None

        # runtime state
        self.next_frame_idx = -1  # -1: extraction hasn't happened yet
        self.container: Optional[docker.models.containers.Container] = None
        self.observer: Optional[Observer] = None
        self.path_queue = queue.SimpleQueue()  # filled by self.observer
        # we keep a list of frames, indexed by frame_idx, because the notifications about new frames
        # may arrive out of order
        self.frame_paths: List[Optional[str]] = []

    def _start_extraction(self) -> None:
        assert self.next_frame_idx == -1

        # use ffmpeg-python to construct the command
        s = ffmpeg.input(str(Path('/input') / self.video_path.name))
        if self.fps > 0:
            s = s.filter('fps', self.fps)
        if self.ffmpeg_filter is not None:
            for key, val in self.ffmpeg_filter.items():
                s = s.filter(key, val)
        # vsync=0: required to apply filter, otherwise ffmpeg pads the output with duplicate frames
        # threads=4: doesn't seem to go faster beyond that
        s = s.output(str(Path('/output') / f'{self.id}_%07d.jpg'), vsync=0, threads=4)
        command = ' '.join([f"'{arg}'" for arg in s.get_args()])  # quote everything to deal with spaces

        class Handler(PatternMatchingEventHandler):
            def __init__(
                    self, pattern: str, output_queue: queue.SimpleQueue, container: docker.models.containers.Container):
                super().__init__(patterns=[pattern], ignore_patterns=None, ignore_directories=True, case_sensitive=True)
                self.output_queue = output_queue
                self.container = container

            def on_closed(self, event: FileSystemEvent) -> None:
                self.output_queue.put(event)
                _logger.debug(f'added {event.src_path} to path_queue (len={self.output_queue.qsize()})')
                # we pause the container if it gets too far ahead of us
                # if self.output_queue.qsize() == 20:
                #     self.container.reload()
                #     if self.container.status == 'running':
                #         _logger.debug(f'pausing container: {self.container.id}')
                #         self.container.pause()

        # start watching for files in tmp_frames_dir
        self.observer = Observer()
        handler = Handler(f'{self.id}_*.jpg', self.path_queue, self.container)
        self.observer.schedule(handler, path=str(Env.get().tmp_frames_dir), recursive=False)
        self.observer.start()

        _logger.debug(f'running ffmpeg: {command}')
        cl = docker.from_env()
        self.container = cl.containers.run(
            'jrottenberg/ffmpeg:4.1-alpine',
            command,
            detach=True,
            remove=False,  # make sure we can reload(), even if it exits beforehand
            volumes={
                self.video_path.parent: {'bind': '/input', 'mode': 'rw'},
                str(Env.get().tmp_frames_dir): {'bind': '/output', 'mode': 'rw'},
            },
            user=os.getuid(),
            group_add=[os.getgid()],
        )
        self.next_frame_idx = 0

    def __iter__(self) -> Iterator[Tuple[int, Path]]:
        return self

    def __next__(self) -> Tuple[int, Path]:
        """
        Returns (frame idx, path to img file).
        """
        if self.next_frame_idx == -1:
            self._start_extraction()
        prev_frame_idx = self.next_frame_idx - 1
        if prev_frame_idx >= 0:
            # try to delete the file
            try:
                os.remove(str(self.frame_paths[prev_frame_idx]))
                _logger.debug(f'removed {self.frame_paths[prev_frame_idx]}')
            except FileNotFoundError as e:
                # nothing to worry about, someone else grabbed it
                pass

        if self.next_frame_idx == self.num_frames:
            self.observer.stop()
            self.observer.join()
            raise StopIteration

        if len(self.frame_paths) < self.next_frame_idx + 1 \
                and (self.num_frames is None or len(self.frame_paths) < self.num_frames):
            # pad frame_paths to simplify the loop
            assert len(self.frame_paths) == self.next_frame_idx
            self.frame_paths.append(None)

        # we need to return the frame at next_frame_idx; make sure we have it
        while self.frame_paths[self.next_frame_idx] is None:
            # check on the container status
            if self.container.status != 'exited':
                self.container.reload()
                if self.container.status == 'exited':
                    # get actual number of frames from stdout
                    log_output = self.container.logs(stdout=True).splitlines()
                    info_line = log_output[-2]
                    # extract frame count with regular expression
                    m = re.search(r'frame=\s*(\d+)', info_line.decode('utf-8'))
                    if m is None:
                        raise RuntimeError('Could not extract frame count from ffmpeg output')
                    self.num_frames = int(m.group(1))
                    _logger.debug(f'container exited, num_frames={self.num_frames}')
                    if self.next_frame_idx == self.num_frames:
                        self.observer.stop()
                        self.observer.join()
                        raise StopIteration
                if self.container.status == 'paused' and self.path_queue.qsize() < 10:
                    # the producer is paused and we're running low on frames: we need to unpause the container
                    _logger.debug(f'unpausing container: {self.container.id}')
                    self.container.unpause()

            _logger.debug(f'waiting for {self.next_frame_idx} (len={self.path_queue.qsize()}, container status={self.container.status}')
            # wait for the next frame to be extracted;
            # we need a timeout to avoid a deadlock situation:
            # the container hasn't exited yet, but we have already returned the last frame
            try:
                event = self.path_queue.get(timeout=1)
                self._add_path(Path(event.src_path))
            except queue.Empty:
                pass

        assert self.frame_paths[self.next_frame_idx] is not None
        result = (self.next_frame_idx, self.frame_paths[self.next_frame_idx])
        self.next_frame_idx += 1
        _logger.debug(f'returning {result}')
        return result

    def _add_path(self, path: Path) -> None:
        """
        Add a path to the frame_paths at the correct index.
        """
        m = self.idx_re.match(path.name)
        if m is None:
            raise RuntimeError(f'Unexpected frame path: {path}')
        idx = int(m.group(1))
        idx -= 1  # ffmpeg starts at 1
        if idx >= len(self.frame_paths):
            self.frame_paths.extend([None] * (idx - len(self.frame_paths) + 1))
        self.frame_paths[idx] = path

    def seek(self, frame_idx: int) -> None:
        """
        Fast-forward to frame idx
        """
        assert frame_idx >= self.next_frame_idx
        if self.next_frame_idx == -1:
            self._start_extraction()
        _logger.debug(f'seeking to frame {frame_idx}')
        while frame_idx < self.next_frame_idx:
            _ = self.__next__()

    def __enter__(self) -> FrameIterator:
        _logger.debug(f'__enter__ {self.id}')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        _logger.debug(f'__exit__ {self.id}')
        return self.close()

    def close(self) -> None:
        _logger.debug(f'closing FrameIterator {self.id}')
        if self.next_frame_idx == -1:
            # nothing to do
            return
        # check if the container is still running; if so, stop it
        self.container.reload()
        if self.container.status != 'exited':
            _logger.debug(f'stopping container {self.container.id}')
            self.container.stop()
        self.container.remove()
        self.observer.stop()
        self.observer.join()

        # remove all remaining files:
        # - whatever is left in path_queue
        # - whatever is left in frame_paths, from next_frame_idx-1 to the end
        while not self.path_queue.empty():
            event = self.path_queue.get()
            try:
                os.remove(event.src_path)
                _logger.debug(f'removed {event.src_path}')
            except FileNotFoundError:
                pass
        # also remove the frame we returned last
        for path in [path for path in self.frame_paths[self.next_frame_idx - 1:] if path is not None]:
            try:
                os.remove(path)
                _logger.debug(f'removed {path}')
            except FileNotFoundError:
                pass


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
    ) -> FrameExtractor:
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

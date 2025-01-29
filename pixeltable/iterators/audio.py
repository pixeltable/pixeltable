import logging
import uuid
import pixeltable.env as env
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import av  # type: ignore[import-untyped]
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class AudioIterator(ComponentIterator):
    """
    Args:
        audio: URL or path of the audio file
        chunk_duration: Target duration for each chunk in seconds to match model requirements.
        overlap: Overlap between chunks in seconds to prevent word splits at chunk boundaries.
        min_chunk_duration: Minimum chunk duration in seconds to match model requirements.
        drop_incomplete_chunks: Drop chunks smaller than min_chunk_duration at the end of the audio.
    """
    # Input parameters
    audio_path: Path
    chunk_duration: float
    overlap: float
    min_chunk_duration: float
    drop_incomplete_chunks: bool

    # audio stream details
    container: av.container.input.InputContainer
    audio_time_base: Fraction
    audio_start_time: int

    # List of chunks to extract each chunk is defined by start time and end time that includes overlap
    chunks_to_extract: Optional[list[tuple[int, int]]]
    # next chunk to extract
    next_pos: int

    def __init__(self, audio: str, *, chunk_duration: Optional[float] = None, overlap: Optional[float] = 0.0,
                 min_chunk_duration: Optional[float] = 0.0, drop_incomplete_chunks: Optional[bool] = False):
        if chunk_duration is None:
            raise excs.Error('chunk_duration may be specified')
        if min_chunk_duration is not None and chunk_duration < min_chunk_duration:
            raise excs.Error('chunk_duration must be at least min_chunk_duration')
        audio_path = Path(audio)
        assert audio_path.exists() and audio_path.is_file()
        self.audio_path = audio_path
        self.container = av.open(str(audio_path))
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.min_chunk_duration = min_chunk_duration
        self.drop_incomplete_chunks = drop_incomplete_chunks
        self.audio_time_base = self.container.streams.audio[0].time_base
        self.audio_start_time = self.container.streams.audio[0].start_time or 0

        total_audio_duration = self.container.streams.audio[0].duration or 0
        pts_chunk_duration = int(self.chunk_duration / self.audio_time_base)
        pts_overlap = int(self.overlap / self.audio_time_base)
        pts_min_chunk_duration = int(self.min_chunk_duration / self.audio_time_base)
        self.chunks_to_extract = []

        for i in range(self.audio_start_time, total_audio_duration, pts_chunk_duration):
            # checks for the last chunk
            if i + pts_chunk_duration >= total_audio_duration:
                duration = total_audio_duration - i
                if pts_min_chunk_duration > 0 and duration < pts_min_chunk_duration and drop_incomplete_chunks:
                    break
                else:
                    self.chunks_to_extract.append((i, total_audio_duration))
                    break
            else:
                self.chunks_to_extract.append((i, min(i + pts_chunk_duration + pts_overlap, total_audio_duration)))

        _logger.debug(f'AudioIterator: path={self.audio_path}  total_audio_duration={total_audio_duration}'
                      f'pts_chunk_duration={pts_chunk_duration} pts_overlap={pts_overlap}'
                      f'chunks_to_extract={self.chunks_to_extract}')
        self.next_pos = 0

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'audio': ts.AudioType(nullable=False),
            'chunk_duration': ts.FloatType(nullable=True),
            'overlap': ts.FloatType(nullable=True),
            'min_chunk_duration': ts.FloatType(nullable=True),
            'drop_incomplete_chunks': ts.BoolType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'chunk_idx': ts.IntType(),
            'start_time_msec': ts.IntType(),
            'end_time_msec': ts.IntType(),
            'duration_msec': ts.IntType(),
            'audio_chunk': ts.AudioType(nullable=True),
        }, ['audio_chunk']

    def __next__(self) -> dict[str, Any]:
        if self.next_pos >= len(self.chunks_to_extract):
            raise StopIteration
        else:
            target_chunk_start = self.chunks_to_extract[self.next_pos][0]
            target_chunk_end = self.chunks_to_extract[self.next_pos][1]
            chunk_start_pts = 0
            chunk_file = self._create_chunk_file()
            output_container = av.open(chunk_file, mode='w')
            input_stream = self.container.streams.audio[0]

            # TODO fix this
            if input_stream.codec_context.name == "mp3float":
                codec_name = "mp3"
            else:
                codec_name = input_stream.codec_context.name
            output_stream = output_container.add_stream(codec_name=codec_name)
            output_stream.codec_context.sample_rate = input_stream.codec_context.sample_rate
            output_stream.codec_context.channels = input_stream.codec_context.channels
            output_stream.time_base = input_stream.time_base

            frame_count = 0
            while True:
                try:
                    frame = next(self.container.decode(audio=0))
                except EOFError:
                    raise StopIteration

                if frame.pts < target_chunk_start:
                    # current frame is behind chunk's start time
                    continue
                frame_end = frame.pts + frame.samples
                # Check that frame's start and end covers start of this chunk
                assert target_chunk_start < frame_end and frame.pts >= target_chunk_start
                if frame_count == 0:
                    # Record start of the first frame
                    chunk_start_pts = frame.pts
                frame_count += 1
                # write frame
                packet = output_stream.encode(frame)
                if packet:
                    output_container.mux(packet)
                if frame_end < target_chunk_end:
                    # we have more data to collect
                    continue
                # record result
                chunk_end_pts = frame_end
                if frame_count > 0:
                    packet = output_stream.encode(None)
                    if packet:
                        output_container.mux(packet)
                    output_container.close()
                result = {
                    'chunk_idx': self.next_pos,
                    'start_time_msec': int(chunk_start_pts * self.audio_time_base * 1000),
                    'end_time_msec': int(chunk_end_pts * self.audio_time_base * 1000),
                    'duration_msec': int((chunk_end_pts - chunk_start_pts) * self.audio_time_base * 1000),
                    'audio_chunk': chunk_file if frame_count > 0 else None
                }
                _logger.debug('audio chunk result: %s', result)
                self.next_pos += 1
                return result

    def _create_chunk_file(self) -> str:
        return str(env.Env.get().tmp_dir / f'{uuid.uuid4()}{self.audio_path.suffix}')

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int) -> None:
        if pos == self.next_pos:
            return  # already there
        # get the start of chunk which is already saved in pts
        target_chunk_start = self.chunks_to_extract[pos][0]
        _logger.debug(f'seeking to presentation time  {target_chunk_start} (at iterator index {pos})')
        self.container.seek(target_chunk_start, backward=True, stream=self.container.streams.audio[0])
        self.next_pos = pos

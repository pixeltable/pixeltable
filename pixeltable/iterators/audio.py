import logging
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

import av  # type: ignore[import-untyped]

import pixeltable.env as env
import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class AudioSplitter(ComponentIterator):
    """
    Iterator over audio data and split it in the chunks of audio clips duration of which is provided in chunk_duration_sec.
    Iterator yields audio chunk as pxt.Audio along with information about start and end time of each audio chunk.
    If there is no audio in the input then no chunks are yielded.

    Args:
        audio: URL or path of the audio file
        chunk_duration_sec: Target duration for each chunk in seconds.
        overlap_sec: Overlap between chunks in seconds.
        min_chunk_duration_sec: Minimum chunk duration in seconds.
        drop_incomplete_chunks: Drop chunks smaller than min_chunk_duration at the end of the audio.
    """

    # Input parameters
    audio_path: Path
    chunk_duration_sec: float
    overlap_sec: float
    min_chunk_duration_sec: float
    drop_incomplete_chunks: bool

    # audio stream details
    container: av.container.input.InputContainer
    audio_time_base: Fraction
    audio_start_time: int

    # List of chunks to extract
    # Each chunk is defined by start and end presentation timestamps in audio file (int)
    chunks_to_extract: Optional[list[tuple[int, int]]]
    # next chunk to extract
    next_pos: int

    __codec_map = {
        'mp3': 'mp3',  # MP3 decoder -> mp3/libmp3lame encoder
        'mp3float': 'mp3',  # MP3float decoder -> mp3 encoder
        'aac': 'aac',  # AAC decoder -> AAC encoder
        'vorbis': 'libvorbis',  # Vorbis decoder -> libvorbis encoder
        'opus': 'libopus',  # Opus decoder -> libopus encoder
        'flac': 'flac',  # FLAC decoder -> FLAC encoder
        'wavpack': 'wavpack',  # WavPack decoder -> WavPack encoder
        'alac': 'alac',  # ALAC decoder -> ALAC encoder
    }

    def __init__(
        self,
        audio: str,
        chunk_duration_sec: float,
        *,
        overlap_sec: float = 0.0,
        min_chunk_duration_sec: float = 0.0,
        drop_incomplete_chunks: bool = False,
    ):
        if chunk_duration_sec <= 0.0:
            raise excs.Error('chunk_duration_sec must be a positive number')
        if min_chunk_duration_sec is not None and chunk_duration_sec < min_chunk_duration_sec:
            raise excs.Error('chunk_duration_sec must be at least min_chunk_duration_sec')
        audio_path = Path(audio)
        assert audio_path.exists() and audio_path.is_file()
        self.audio_path = audio_path
        self.container = av.open(str(audio_path))
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.drop_incomplete_chunks = drop_incomplete_chunks
        self.audio_time_base = self.container.streams.audio[0].time_base
        self.audio_start_time = self.container.streams.audio[0].start_time or 0

        total_audio_duration = self.container.streams.audio[0].duration or 0
        pts_chunk_duration = int(self.chunk_duration_sec / self.audio_time_base)
        pts_overlap = int(self.overlap_sec / self.audio_time_base)
        pts_min_chunk_duration = int(self.min_chunk_duration_sec / self.audio_time_base)
        self.chunks_to_extract = self.build_chunks(
            self.audio_start_time,
            total_audio_duration,
            pts_chunk_duration,
            pts_overlap,
            pts_min_chunk_duration,
            self.drop_incomplete_chunks,
        )
        _logger.debug(
            f'AudioIterator: path={self.audio_path} total_audio_duration={total_audio_duration}'
            f' pts_chunk_duration={pts_chunk_duration} pts_overlap={pts_overlap}'
            f' chunks_to_extract={self.chunks_to_extract}'
        )
        self.next_pos = 0

    @classmethod
    def build_chunks(
        cls, start_time, total_duration, chunk_duration, overlap, min_chunk_duration, drop_incomplete_chunks
    ) -> list[tuple[int, int]]:
        chunks_to_extract: list[tuple[int, int]] = []
        for i in range(start_time, total_duration, chunk_duration):
            # checks for the last chunk
            if i + chunk_duration + overlap >= total_duration:
                duration = total_duration - i
                if not (drop_incomplete_chunks and min_chunk_duration > 0 and duration < min_chunk_duration):
                    chunks_to_extract.append((i, total_duration))
                break
            else:
                chunks_to_extract.append((i, i + chunk_duration + overlap))
        return chunks_to_extract

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'audio': ts.AudioType(nullable=False),
            'chunk_duration_sec': ts.FloatType(nullable=True),
            'overlap_sec': ts.FloatType(nullable=True),
            'min_chunk_duration_sec': ts.FloatType(nullable=True),
            'drop_incomplete_chunks': ts.BoolType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'chunk_idx': ts.IntType(),
            'start_time_sec': ts.FloatType(),
            'end_time_sec': ts.FloatType(),
            'duration_sec': ts.FloatType(),
            'audio_chunk': ts.AudioType(nullable=True),
        }, ['audio_chunk']

    def __next__(self) -> dict[str, Any]:
        if self.next_pos >= len(self.chunks_to_extract):
            raise StopIteration
        else:
            target_chunk_start, target_chunk_end = self.chunks_to_extract[self.next_pos]
            chunk_start_pts = 0
            chunk_end_pts = 0
            chunk_file = self.__create_chunk_file()
            output_container = av.open(chunk_file, mode='w')
            input_stream = self.container.streams.audio[0]
            codec_name = AudioSplitter.__codec_map.get(input_stream.codec_context.name, input_stream.codec_context.name)
            output_stream = output_container.add_stream(codec_name=codec_name)
            output_stream.codec_context.sample_rate = input_stream.codec_context.sample_rate
            output_stream.codec_context.channels = input_stream.codec_context.channels
            output_stream.time_base = input_stream.time_base

            frame_count = 0
            # Since frames don't align with chunk boundaries, we may have read an extra frame in previous iteration
            # Seek to the nearest frame in stream at current chunk start time
            self.container.seek(target_chunk_start, backward=True, stream=self.container.streams.audio[0])
            while True:
                try:
                    frame = next(self.container.decode(audio=0))
                except EOFError as e:
                    raise excs.Error(f'Failed to read audio file `{self.audio_path}`, error `{e}`')
                except StopIteration:
                    # no more frames to scan
                    break
                if frame.pts < target_chunk_start:
                    # Current frame is behind chunk's start time, always get frame next to chunk's start time
                    continue
                if frame.pts >= target_chunk_end:
                    # Frame has crossed the chunk boundary, it should be picked up by next chunk, throw away the current frame
                    break
                frame_end = frame.pts + frame.samples
                if frame_count == 0:
                    # Record start of the first frame
                    chunk_start_pts = frame.pts
                # Write frame to output container
                frame_count += 1
                # If encode returns packets, write them to output container. Some encoders will buffer the frames.
                output_container.mux(output_stream.encode(frame))
                # record this frame's end as chunks end
                chunk_end_pts = frame_end
                # Check if frame's end has crossed the chunk boundary
                if frame_end >= target_chunk_end:
                    break

            # record result
            if frame_count > 0:
                # flush encoder
                output_container.mux(output_stream.encode(None))
                output_container.close()

            result = {
                'chunk_idx': self.next_pos,
                'start_time_sec': round(float(chunk_start_pts * self.audio_time_base), 4),
                'end_time_sec': round(float(chunk_end_pts * self.audio_time_base), 4),
                'duration_sec': round(float((chunk_end_pts - chunk_start_pts) * self.audio_time_base), 4),
                'audio_chunk': chunk_file if frame_count > 0 else None,
            }
            _logger.debug('audio chunk result: %s', result)
            self.next_pos += 1
            return result

    def __create_chunk_file(self) -> str:
        return str(env.Env.get().tmp_dir / f'{uuid.uuid4()}{self.audio_path.suffix}')

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int) -> None:
        if pos == self.next_pos:
            return  # already there
        # get the start of chunk which is already saved in presentation time
        target_chunk_start = self.chunks_to_extract[pos][0]
        _logger.debug(f'seeking to presentation time {target_chunk_start} (at iterator index {pos})')
        self.container.seek(target_chunk_start, backward=True, stream=self.container.streams.audio[0])
        self.next_pos = pos

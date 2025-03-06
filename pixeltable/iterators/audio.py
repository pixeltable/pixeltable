import logging
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar, Optional

import av

from pixeltable import env, exceptions as excs, type_system as ts

from .base import ComponentIterator

_logger = logging.getLogger('pixeltable')


class AudioSplitter(ComponentIterator):
    """
    Iterator over chunks of an audio file. The audio file is split into smaller chunks,
    where the duration of each chunk is determined by chunk_duration_sec.
    The iterator yields audio chunks as pxt.Audio, along with the start and end time of each chunk.
    If the input contains no audio, no chunks are yielded.

    Args:
        chunk_duration_sec: Audio chunk duration in seconds
        overlap_sec: Overlap between consecutive chunks in seconds.
        min_chunk_duration_sec: Drop the last chunk if it is smaller than min_chunk_duration_sec
    """

    # Input parameters
    audio_path: Path
    chunk_duration_sec: float
    overlap_sec: float

    # audio stream details
    container: av.container.input.InputContainer
    audio_time_base: Fraction  # seconds per presentation time

    # List of chunks to extract
    # Each chunk is defined by start and end presentation timestamps in audio file (int)
    chunks_to_extract_in_pts: Optional[list[tuple[int, int]]]
    # next chunk to extract
    next_pos: int

    __codec_map: ClassVar[dict[str, str]] = {
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
        self, audio: str, chunk_duration_sec: float, *, overlap_sec: float = 0.0, min_chunk_duration_sec: float = 0.0
    ):
        if chunk_duration_sec <= 0.0:
            raise excs.Error('chunk_duration_sec must be a positive number')
        if chunk_duration_sec < min_chunk_duration_sec:
            raise excs.Error('chunk_duration_sec must be at least min_chunk_duration_sec')
        if overlap_sec >= chunk_duration_sec:
            raise excs.Error('overlap_sec must be less than chunk_duration_sec')
        audio_path = Path(audio)
        assert audio_path.exists() and audio_path.is_file()
        self.audio_path = audio_path
        self.next_pos = 0
        self.container = av.open(str(audio_path))
        if len(self.container.streams.audio) == 0:
            # No audio stream
            return
        self.chunk_duration_sec = chunk_duration_sec
        self.overlap_sec = overlap_sec
        self.min_chunk_duration_sec = min_chunk_duration_sec
        self.audio_time_base = self.container.streams.audio[0].time_base

        audio_start_time_pts = self.container.streams.audio[0].start_time or 0
        audio_start_time_sec = float(audio_start_time_pts * self.audio_time_base)
        total_audio_duration_pts = self.container.streams.audio[0].duration or 0
        total_audio_duration_sec = float(total_audio_duration_pts * self.audio_time_base)

        self.chunks_to_extract_in_pts = [
            (round(start / self.audio_time_base), round(end / self.audio_time_base))
            for (start, end) in self.build_chunks(
                audio_start_time_sec, total_audio_duration_sec, chunk_duration_sec, overlap_sec, min_chunk_duration_sec
            )
        ]
        _logger.debug(
            f'AudioIterator: path={self.audio_path} total_audio_duration_pts={total_audio_duration_pts} '
            f'chunks_to_extract_in_pts={self.chunks_to_extract_in_pts}'
        )

    @classmethod
    def build_chunks(
        cls,
        start_time_sec: float,
        total_duration_sec: float,
        chunk_duration_sec: float,
        overlap_sec: float,
        min_chunk_duration_sec: float,
    ) -> list[tuple[float, float]]:
        chunks_to_extract_in_sec: list[tuple[float, float]] = []
        current_pos = start_time_sec
        end_time = start_time_sec + total_duration_sec
        while current_pos < end_time:
            chunk_start = current_pos
            chunk_end = min(chunk_start + chunk_duration_sec, end_time)
            chunks_to_extract_in_sec.append((chunk_start, chunk_end))
            if chunk_end >= end_time:
                break
            current_pos = chunk_end - overlap_sec
        # If the last chunk is smaller than min_chunk_duration_sec then drop the last chunk from the list
        if (
            len(chunks_to_extract_in_sec) > 0
            and (chunks_to_extract_in_sec[-1][1] - chunks_to_extract_in_sec[-1][0]) < min_chunk_duration_sec
        ):
            return chunks_to_extract_in_sec[:-1]  # return all but the last chunk
        return chunks_to_extract_in_sec

    @classmethod
    def input_schema(cls) -> dict[str, ts.ColumnType]:
        return {
            'audio': ts.AudioType(nullable=False),
            'chunk_duration_sec': ts.FloatType(nullable=True),
            'overlap_sec': ts.FloatType(nullable=True),
            'min_chunk_duration_sec': ts.FloatType(nullable=True),
        }

    @classmethod
    def output_schema(cls, *args: Any, **kwargs: Any) -> tuple[dict[str, ts.ColumnType], list[str]]:
        return {
            'start_time_sec': ts.FloatType(),
            'end_time_sec': ts.FloatType(),
            'audio_chunk': ts.AudioType(nullable=True),
        }, []

    def __next__(self) -> dict[str, Any]:
        if self.next_pos >= len(self.chunks_to_extract_in_pts):
            raise StopIteration
        target_chunk_start, target_chunk_end = self.chunks_to_extract_in_pts[self.next_pos]
        chunk_start_pts = 0
        chunk_end_pts = 0
        chunk_file = str(env.Env.get().tmp_dir / f'{uuid.uuid4()}{self.audio_path.suffix}')
        output_container = av.open(chunk_file, mode='w')
        input_stream = self.container.streams.audio[0]
        codec_name = AudioSplitter.__codec_map.get(input_stream.codec_context.name, input_stream.codec_context.name)
        output_stream = output_container.add_stream(codec_name, rate=input_stream.codec_context.sample_rate)
        assert isinstance(output_stream, av.audio.stream.AudioStream)
        frame_count = 0
        # Since frames don't align with chunk boundaries, we may have read an extra frame in previous iteration
        # Seek to the nearest frame in stream at current chunk start time
        self.container.seek(target_chunk_start, backward=True, stream=self.container.streams.audio[0])
        while True:
            try:
                frame = next(self.container.decode(audio=0))
            except EOFError as e:
                raise excs.Error(f"Failed to read audio file '{self.audio_path}': {e}") from e
            except StopIteration:
                # no more frames to scan
                break
            if frame.pts < target_chunk_start:
                # Current frame is behind chunk's start time, always get frame next to chunk's start time
                continue
            if frame.pts >= target_chunk_end:
                # Frame has crossed the chunk boundary, it should be picked up by next chunk, throw away
                # the current frame
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
                'start_time_sec': round(float(chunk_start_pts * self.audio_time_base), 4),
                'end_time_sec': round(float(chunk_end_pts * self.audio_time_base), 4),
                'audio_chunk': chunk_file if frame_count > 0 else None,
            }
            _logger.debug('audio chunk result: %s', result)
            self.next_pos += 1
            return result
        else:
            # It's possible that there are no frames in the range of the last chunk, stop the iterator in this case.
            # Note that start_time points at the first frame so case applies only for the last chunk
            assert self.next_pos == len(self.chunks_to_extract_in_pts) - 1
            self.next_pos += 1
            raise StopIteration

    def close(self) -> None:
        self.container.close()

    def set_pos(self, pos: int) -> None:
        pass

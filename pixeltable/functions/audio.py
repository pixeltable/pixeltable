"""
Pixeltable UDFs for `AudioType`.
"""

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import av
import numpy as np

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable import exceptions as excs
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

_logger = logging.getLogger('pixeltable')


@pxt.udf(is_method=True)
def get_metadata(audio: pxt.Audio) -> dict:
    """
    Gets various metadata associated with an audio file and returns it as a dictionary.

    Args:
        audio: The audio to get metadata for.

    Returns:
        A `dict` such as the following:

            ```json
            {
                'size': 2568827,
                'streams': [
                    {
                        'type': 'audio',
                        'frames': 0,
                        'duration': 2646000,
                        'metadata': {},
                        'time_base': 2.2675736961451248e-05,
                        'codec_context': {
                            'name': 'flac',
                            'profile': None,
                            'channels': 1,
                            'codec_tag': '\\x00\\x00\\x00\\x00',
                        },
                        'duration_seconds': 60.0,
                    }
                ],
                'bit_rate': 342510,
                'metadata': {'encoder': 'Lavf61.1.100'},
                'bit_exact': False,
            }
            ```

    Examples:
        Extract metadata for files in the `audio_col` column of the table `tbl`:

        >>> tbl.select(tbl.audio_col.get_metadata()).collect()
    """
    return av_utils.get_metadata(audio)


@pxt.udf()
def encode_audio(
    audio_data: pxt.Array[pxt.Float], *, input_sample_rate: int, format: str, output_sample_rate: int | None = None
) -> pxt.Audio:
    """
    Encodes an audio clip represented as an array into a specified audio format.

    Parameters:
        audio_data: An array of sampled amplitudes. The accepted array shapes are `(N,)` or `(1, N)` for mono audio
            or `(2, N)` for stereo.
        input_sample_rate: The sample rate of the input audio data.
        format: The desired output audio format. The supported formats are 'wav', 'mp3', 'flac', and 'mp4'.
        output_sample_rate: The desired sample rate for the output audio. Defaults to the input sample rate if
            unspecified.

    Examples:
        Add a computed column with encoded FLAC audio files to a table with audio data (as arrays of floats) and sample
        rates:

        >>> t.add_computed_column(
        ...     audio_file=encode_audio(
        ...         t.audio_data, input_sample_rate=t.sample_rate, format='flac'
        ...     )
        ... )
    """
    if format not in av_utils.AUDIO_FORMATS:
        raise pxt.Error(f'Only the following formats are supported: {av_utils.AUDIO_FORMATS.keys()}')
    if output_sample_rate is None:
        output_sample_rate = input_sample_rate

    codec, ext = av_utils.AUDIO_FORMATS[format]
    output_path = str(TempStore.create_path(extension=f'.{ext}'))

    match audio_data.shape:
        case (_,):
            # Mono audio as 1D array, reshape for pyav
            layout = 'mono'
            audio_data_transformed = audio_data[None, :]
        case (1, _):
            # Mono audio as 2D array, simply reshape and transpose the input for pyav
            layout = 'mono'
            audio_data_transformed = audio_data.reshape(-1, 1).transpose()
        case (2, _):
            # Stereo audio. Input layout: [[L0, L1, L2, ...],[R0, R1, R2, ...]],
            # pyav expects: [L0, R0, L1, R1, L2, R2, ...]
            layout = 'stereo'
            audio_data_transformed = np.empty(audio_data.shape[1] * 2, dtype=audio_data.dtype)
            audio_data_transformed[0::2] = audio_data[0]
            audio_data_transformed[1::2] = audio_data[1]
            audio_data_transformed = audio_data_transformed.reshape(1, -1)
        case _:
            raise pxt.Error(
                f'Supported input array shapes are (N,), (1, N) for mono and (2, N) for stereo, got {audio_data.shape}'
            )

    with av.open(output_path, mode='w') as output_container:
        stream = output_container.add_stream(codec, rate=output_sample_rate)
        assert isinstance(stream, av.AudioStream)

        frame = av.AudioFrame.from_ndarray(audio_data_transformed, format='flt', layout=layout)
        frame.sample_rate = input_sample_rate

        for packet in stream.encode(frame):
            output_container.mux(packet)
        for packet in stream.encode():
            output_container.mux(packet)

        return output_path


class AudioSegment(TypedDict):
    segment_start: float
    segment_end: float
    audio_segment: pxt.Audio | None


@pxt.iterator
class audio_splitter(pxt.PxtIterator[AudioSegment]):
    """
    Iterator over segments of an audio file. The audio file is split into smaller segments,
    where the duration of each segment is determined by `duration`.

    If the input contains no audio, no segments are yielded.

    __Outputs__:

        One row per audio segment, with the following columns:

        - `segment_start` (`pxt.Float`): Start time of the audio segment in seconds
        - `segment_end` (`pxt.Float`): End time of the audio segment in seconds
        - `audio_segment` (`pxt.Audio | None`): The audio content of the segment

    Args:
        duration: Audio segment duration in seconds
        overlap: Overlap between consecutive segments in seconds
        min_segment_duration: Drop the last segment if it is smaller than `min_segment_duration`

    Examples:
        This example assumes an existing table `tbl` with a column `audio` of type `pxt.Audio`.

        Create a view that splits all audio files into segments of 30 seconds with 5 seconds overlap:

        >>> pxt.create_view(
        ...     'audio_segments',
        ...     tbl,
        ...     iterator=audio_splitter(tbl.audio, duration=30.0, overlap=5.0),
        ... )
    """

    audio_path: Path
    segment_duration: float
    overlap: float
    min_segment_duration: float

    # audio stream details
    container: av.container.input.InputContainer
    audio_time_base: Fraction  # seconds per presentation time

    # List of segments to extract
    # Each segment is defined by start and end presentation timestamps in audio file (int)
    segments_to_extract_in_pts: list[tuple[int, int]] | None

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

    def __init__(self, audio: pxt.Audio, duration: float, *, overlap: float = 0.0, min_segment_duration: float = 0.0):
        assert duration > 0.0
        assert duration >= min_segment_duration
        assert overlap < duration
        audio_path = Path(audio)
        assert audio_path.exists() and audio_path.is_file()
        self.audio_path = audio_path
        self.next_pos = 0
        self.container = av.open(str(audio_path))
        if len(self.container.streams.audio) == 0:
            # No audio stream
            return
        self.segment_duration = duration
        self.overlap = overlap
        self.min_segment_duration = min_segment_duration
        self.audio_time_base = self.container.streams.audio[0].time_base

        audio_start_time_pts = self.container.streams.audio[0].start_time or 0
        audio_start_time = float(audio_start_time_pts * self.audio_time_base)
        total_audio_duration_pts = self.container.streams.audio[0].duration or 0
        total_audio_duration = float(total_audio_duration_pts * self.audio_time_base)

        self.segments_to_extract_in_pts = [
            (round(start / self.audio_time_base), round(end / self.audio_time_base))
            for (start, end) in self.build_segments(
                audio_start_time, total_audio_duration, duration, overlap, min_segment_duration
            )
        ]
        _logger.debug(
            f'AudioIterator: path={self.audio_path} total_audio_duration_pts={total_audio_duration_pts} '
            f'segments_to_extract_in_pts={self.segments_to_extract_in_pts}'
        )

    @classmethod
    def build_segments(
        cls,
        start_time: float,
        total_duration: float,
        segment_duration: float,
        overlap: float,
        min_segment_duration: float,
    ) -> list[tuple[float, float]]:
        segments_to_extract_in: list[tuple[float, float]] = []
        current_pos = start_time
        end_time = start_time + total_duration
        while current_pos < end_time:
            segment_start = current_pos
            segment_end = min(segment_start + segment_duration, end_time)
            segments_to_extract_in.append((segment_start, segment_end))
            if segment_end >= end_time:
                break
            current_pos = segment_end - overlap
        # If the last segment is smaller than min_segment_duration then drop the last segment from the list
        if (
            len(segments_to_extract_in) > 0
            and (segments_to_extract_in[-1][1] - segments_to_extract_in[-1][0]) < min_segment_duration
        ):
            return segments_to_extract_in[:-1]  # return all but the last segment
        return segments_to_extract_in

    def __next__(self) -> AudioSegment:
        if self.next_pos >= len(self.segments_to_extract_in_pts):
            raise StopIteration
        target_segment_start, target_segment_end = self.segments_to_extract_in_pts[self.next_pos]
        segment_start_pts = 0
        segment_end_pts = 0
        segment_file = str(TempStore.create_path(extension=self.audio_path.suffix))
        output_container = av.open(segment_file, mode='w')
        input_stream = self.container.streams.audio[0]
        codec_name = self.__codec_map.get(input_stream.codec_context.name, input_stream.codec_context.name)
        output_stream = output_container.add_stream(codec_name, rate=input_stream.codec_context.sample_rate)
        assert isinstance(output_stream, av.audio.stream.AudioStream)
        frame_count = 0
        # Since frames don't align with segment boundaries, we may have read an extra frame in previous iteration
        # Seek to the nearest frame in stream at current segment start time
        self.container.seek(target_segment_start, backward=True, stream=self.container.streams.audio[0])
        while True:
            try:
                frame = next(self.container.decode(audio=0))
            except EOFError as e:
                raise excs.Error(f"Failed to read audio file '{self.audio_path}': {e}") from e
            except StopIteration:
                # no more frames to scan
                break
            if frame.pts < target_segment_start:
                # Current frame is behind segment's start time, always get frame next to segment's start time
                continue
            if frame.pts >= target_segment_end:
                # Frame has crossed the segment boundary, it should be picked up by next segment, throw away
                # the current frame
                break
            frame_end = frame.pts + frame.samples
            if frame_count == 0:
                # Record start of the first frame
                segment_start_pts = frame.pts
            # Write frame to output container
            frame_count += 1
            # If encode returns packets, write them to output container. Some encoders will buffer the frames.
            output_container.mux(output_stream.encode(frame))
            # record this frame's end as segments end
            segment_end_pts = frame_end
            # Check if frame's end has crossed the segment boundary
            if frame_end >= target_segment_end:
                break

        # record result
        if frame_count > 0:
            # flush encoder
            output_container.mux(output_stream.encode(None))
            output_container.close()
            result: AudioSegment = {
                'segment_start': round(float(segment_start_pts * self.audio_time_base), 4),
                'segment_end': round(float(segment_end_pts * self.audio_time_base), 4),
                'audio_segment': segment_file if frame_count > 0 else None,
            }
            _logger.debug('audio segment result: %s', result)
            self.next_pos += 1
            return result
        else:
            # It's possible that there are no frames in the range of the last segment, stop the iterator in this case.
            # Note that start_time points at the first frame so case applies only for the last segment
            assert self.next_pos == len(self.segments_to_extract_in_pts) - 1
            self.next_pos += 1
            raise StopIteration

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        duration = bound_args.get('duration')
        overlap = bound_args.get('overlap')
        min_segment_duration = bound_args.get('min_segment_duration')

        if duration is not None and duration <= 0.0:
            raise excs.Error('`duration` must be a positive number')
        if duration is not None and min_segment_duration is not None and duration < min_segment_duration:
            raise excs.Error('`duration` must be at least `min_segment_duration`')
        if duration is not None and overlap is not None and overlap >= duration:
            raise excs.Error('`overlap` must be strictly less than `duration`')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

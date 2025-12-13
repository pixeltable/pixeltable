"""
Pixeltable UDFs for `AudioType`.
"""

from typing import Any

import av
import numpy as np

import pixeltable as pxt
import pixeltable.utils.av as av_utils
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore


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


def audio_splitter(
    audio: Any, chunk_duration_sec: float, *, overlap_sec: float = 0.0, min_chunk_duration_sec: float = 0.0
) -> tuple[type[pxt.iterators.ComponentIterator], dict[str, Any]]:
    """
    Iterator over chunks of an audio file. The audio file is split into smaller chunks,
    where the duration of each chunk is determined by chunk_duration_sec.
    The iterator yields audio chunks as pxt.Audio, along with the start and end time of each chunk.
    If the input contains no audio, no chunks are yielded.

    Args:
        chunk_duration_sec: Audio chunk duration in seconds
        overlap_sec: Overlap between consecutive chunks in seconds
        min_chunk_duration_sec: Drop the last chunk if it is smaller than min_chunk_duration_sec

    Examples:
        This example assumes an existing table `tbl` with a column `audio` of type `pxt.Audio`.

        Create a view that splits all audio files into chunks of 30 seconds with 5 seconds overlap:

        >>> pxt.create_view(
        ...     'audio_chunks',
        ...     tbl,
        ...     iterator=audio_splitter(tbl.audio, chunk_duration_sec=30.0, overlap_sec=5.0)
        ... )
    """
    kwargs: dict[str, Any] = {}
    if overlap_sec != 0.0:
        kwargs['overlap_sec'] = overlap_sec
    if min_chunk_duration_sec != 0.0:
        kwargs['min_chunk_duration_sec'] = min_chunk_duration_sec
    return pxt.iterators.AudioSplitter._create(audio=audio, chunk_duration_sec=chunk_duration_sec, **kwargs)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

import av
import numpy as np

import pixeltable as pxt
from pixeltable.utils.code import local_public_names
from pixeltable.utils.local_store import TempStore

# format -> (codec, file extension)
audio_formats: dict[str, tuple[str, str]] = {
    'wav': ('pcm_s16le', 'wav'),
    'mp3': ('libmp3lame', 'mp3'),
    'flac': ('flac', 'flac'),
    'mp4': ('aac', 'm4a'),
}


@pxt.udf(is_method=True)
def to_audio(
    audio_data: pxt.Array, *, input_sample_rate: int, format: str, output_sample_rate: int | None = None
) -> pxt.Audio:
    """
    Encodes an audio clip represented as an array into a specified audio format.

    Parameters:
        audio_data: An array representing the audio data. The shape should be (1, N) for mono audio or (2, N) for
        stereo.
        input_sample_rate: The sample rate of the input audio data.
        format: The desired output audio format. The supported formats are "wav", "mp3", "flac", and "mp4".
        output_sample_rate: The desired sample rate for the output audio. Defaults to the input sample rate if
        unspecified.

    Example:
        update_status = t.add_computed_column(
            audio_file=to_audio(
                t.audio.array.astype(pxt.Array), input_sample_rate=t.audio.sampling_rate.astype(pxt.Int), format='flac'
            ),
        )
    """
    if format not in audio_formats:
        raise pxt.Error(f'Only the following formats are supported: {audio_formats.keys}')
    if output_sample_rate is None:
        output_sample_rate = input_sample_rate

    codec, ext = audio_formats[format]
    output_path = str(TempStore.create_path(extension=f'.{ext}'))

    match audio_data.shape[0]:
        case 1:
            # Mono audio, simply reshape and transpose the input for pyav
            layout = 'mono'
            audio_data_transformed = audio_data.reshape(-1, 1).transpose()
        case 2:
            # Stereo audio. Input layout: [[L0, L1, L2, ...],[R0, R1, R2, ...]],
            # pyav expects: [L0, R0, L1, R1, L2, R2, ...]
            layout = 'stereo'
            audio_data_transformed = np.empty(audio_data.shape[1] * 2, dtype=audio_data.dtype)
            audio_data_transformed[0::2] = audio_data[0]
            audio_data_transformed[1::2] = audio_data[1]
            audio_data_transformed = audio_data_transformed.reshape(1, -1)
        case _:
            raise pxt.Error(
                f'Supported input array shapes are (1, N) for mono and (2, N) for stereo, got {audio_data.shape}'
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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

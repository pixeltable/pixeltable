"""
Pixeltable [UDF](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wraps the OpenAI Whisper library.

This UDF will cause Pixeltable to invoke the relevant model locally. In order to use it, you must
first `pip install openai-whisper`.
"""

from typing import TYPE_CHECKING, Optional, Sequence

import pixeltable as pxt
from pixeltable.env import Env

if TYPE_CHECKING:
    from whisper import Whisper  # type: ignore[import-untyped]


@pxt.udf
def transcribe(
    audio: pxt.Audio,
    *,
    model: str,
    temperature: Optional[Sequence[float]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = '"\'“¿([{-',
    append_punctuations: str = '"\'.。,，!！?？:：”)]}、',  # noqa: RUF001
    decode_options: Optional[dict] = None,
) -> dict:
    """
    Transcribe an audio file using Whisper.

    This UDF runs a transcription model _locally_ using the Whisper library,
    equivalent to the Whisper `transcribe` function, as described in the
    [Whisper library documentation](https://github.com/openai/whisper).

    __Requirements:__

    - `pip install openai-whisper`

    Args:
        audio: The audio file to transcribe.
        model: The name of the model to use for transcription.

    Returns:
        A dictionary containing the transcription and various other metadata.

    Examples:
        Add a computed column that applies the model `base.en` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl.add_computed_column(result=transcribe(tbl.audio, model='base.en'))
    """
    Env.get().require_package('whisper')
    Env.get().require_package('torch')
    import torch

    if decode_options is None:
        decode_options = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = _lookup_model(model, device)
    result = model.transcribe(
        audio,
        temperature=tuple(temperature),
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=initial_prompt,
        word_timestamps=word_timestamps,
        prepend_punctuations=prepend_punctuations,
        append_punctuations=append_punctuations,
        **decode_options,
    )
    return result


def _lookup_model(model_id: str, device: str) -> 'Whisper':
    import whisper

    key = (model_id, device)
    if key not in _model_cache:
        model = whisper.load_model(model_id, device)
        _model_cache[key] = model
    return _model_cache[key]


_model_cache: dict[tuple[str, str], 'Whisper'] = {}

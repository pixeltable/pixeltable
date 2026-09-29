"""
Pixeltable UDFs
that wraps the OpenAI Whisper library.

This UDF will cause Pixeltable to invoke the relevant model locally. In order to use it, you must
first `pip install openai-whisper`.
"""

import threading
from typing import TYPE_CHECKING, NotRequired, Sequence, TypedDict

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from whisper import Whisper  # type: ignore[import-untyped]


class WhisperWord(TypedDict):
    """One word of a transcription segment, with its timing."""

    word: str
    """The word's text, including any leading space and attached punctuation."""
    start: float
    """Word start, in seconds from the start of the audio."""
    end: float
    """Word end, in seconds from the start of the audio."""
    probability: float
    """Mean probability of the word's tokens."""


class WhisperSegment(TypedDict):
    """One segment of a transcription, with its timing and decoding statistics."""

    id: int
    """Index of the segment in the transcription's `segments` list."""
    seek: int
    """Start of the segment's decoding window, in 10 ms frames."""
    start: float
    """Segment start, in seconds from the start of the audio."""
    end: float
    """Segment end, in seconds from the start of the audio."""
    text: str
    """The segment's text."""
    tokens: list[int]
    """Token IDs of the segment; can include timestamp tokens."""
    temperature: float
    """Sampling temperature of the segment's decoding window."""
    avg_logprob: float
    """Average log probability of the tokens in the segment's decoding window."""
    compression_ratio: float
    """Compression ratio of the text in the segment's decoding window; a high value indicates repetitive output."""
    no_speech_prob: float
    """Probability that the segment's decoding window contains no speech."""
    words: NotRequired[list[WhisperWord]]
    """Timings of the segment's words; present only when `word_timestamps=True`."""


class WhisperTranscription(TypedDict):
    """Output of [`transcribe()`][pixeltable.functions.whisper.transcribe]."""

    text: str
    """The full transcription."""
    segments: list[WhisperSegment]
    """The transcription's segments, in order."""
    language: str
    """Code of the transcription's language, such as `en`."""


@pxt.udf
def transcribe(
    audio: pxt.Audio,
    *,
    model: str,
    temperature: Sequence[float] | None = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float | None = 2.4,
    logprob_threshold: float | None = -1.0,
    no_speech_threshold: float | None = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: str | None = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = '"\'“¿([{-',
    append_punctuations: str = '"\'.。,，!！?？:：”)]}、',  # noqa: RUF001
    decode_options: dict | None = None,
) -> WhisperTranscription:
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
        A [`WhisperTranscription`][pixeltable.functions.whisper.WhisperTranscription] dictionary with the
        transcription's text, segments, and language.

    Examples:
        Add a computed column that applies the model `base.en` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl.add_computed_column(result=transcribe(tbl.audio, model='base.en'))

        Add a `String` column with the transcription's text:

        >>> tbl.add_computed_column(text=tbl.result.text)
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
    with _cache_lock:
        if key not in _model_cache:
            model = whisper.load_model(model_id, device)
            _model_cache[key] = model
        return _model_cache[key]


# guards the cache below; held across model loads so a cache miss never loads twice
_cache_lock = threading.Lock()
_model_cache: dict[tuple[str, str], 'Whisper'] = {}


_class_names = ['WhisperTranscription', 'WhisperSegment', 'WhisperWord']
__all__ = local_public_names(__name__) + _class_names


def __dir__() -> list[str]:
    return __all__

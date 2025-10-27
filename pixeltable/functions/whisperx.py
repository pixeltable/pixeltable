"""WhisperX audio transcription and diarization functions."""

from typing import TYPE_CHECKING, Any

import numpy as np

import pixeltable as pxt
from pixeltable.config import Config
from pixeltable.functions.util import resolve_torch_device
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from transformers import Wav2Vec2Model
    from whisperx.asr import FasterWhisperPipeline  # type: ignore[import-untyped]
    from whisperx.diarize import DiarizationPipeline  # type: ignore[import-untyped]


@pxt.udf
def transcribe(
    audio: pxt.Audio,
    *,
    model: str,
    diarize: bool = False,
    compute_type: str | None = None,
    language: str | None = None,
    task: str | None = None,
    chunk_size: int | None = None,
    alignment_model_name: str | None = None,
    interpolate_method: str | None = None,
    return_char_alignments: bool | None = None,
    diarization_model_name: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> dict:
    """
    Transcribe an audio file using WhisperX.

    This UDF runs a transcription model _locally_ using the WhisperX library,
    equivalent to the WhisperX `transcribe` function, as described in the
    [WhisperX library documentation](https://github.com/m-bain/whisperX).

    If `diarize=True`, then speaker diarization will also be performed. Several of the UDF parameters are only valid if
    `diarize=True`, as documented in the parameters list below.

    __Requirements:__

    - `pip install whisperx`

    Args:
        audio: The audio file to transcribe.
        model: The name of the model to use for transcription.
        diarize: Whether to perform speaker diarization.
        compute_type: The compute type to use for the model (e.g., `'int8'`, `'float16'`). If `None`,
            defaults to `'float16'` on CUDA devices and `'int8'` otherwise.
        language: The language code for the transcription (e.g., `'en'` for English).
        task: The task to perform (e.g., `'transcribe'` or `'translate'`). Defaults to `'transcribe'`.
        chunk_size: The size of the audio chunks to process, in seconds. Defaults to `30`.
        alignment_model_name: The name of the alignment model to use. If `None`, uses the default model for the given
            language. Only valid if `diarize=True`.
        interpolate_method: The method to use for interpolation of the alignment results. If not specified, uses the
            WhisperX default (`'nearest'`). Only valid if `diarize=True`.
        return_char_alignments: Whether to return character-level alignments. Defaults to `False`.
            Only valid if `diarize=True`.
        diarization_model_name: The name of the diarization model to use. Defaults to
            `pyannote/speaker-diarization-3.1`. Only valid if `diarize=True`.
        num_speakers: The number of speakers to expect in the audio. By default, the model with try to detect the
            number of speakers. Only valid if `diarize=True`.
        min_speakers: If specified, the minimum number of speakers to expect in the audio.
            Only valid if `diarize=True`.
        max_speakers: If specified, the maximum number of speakers to expect in the audio.
            Only valid if `diarize=True`.

    Returns:
        A dictionary containing the audio transcription, diarization (if enabled), and various other metadata.

    Examples:
        Add a computed column that applies the model `tiny.en` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl.add_computed_column(result=transcribe(tbl.audio, model='tiny.en'))

        Add a computed column that applies the model `tiny.en` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`, with speaker diarization enabled, expecting at least 2 speakers:

        >>> tbl.add_computed_column(
        ...     result=transcribe(
        ...         tbl.audio, model='tiny.en', diarize=True, min_speakers=2
        ...     )
        ... )
    """
    import whisperx  # type: ignore[import-untyped]

    if not diarize:
        args = locals()
        for param in (
            'alignment_model_name',
            'interpolate_method',
            'return_char_alignments',
            'diarization_model_name',
            'num_speakers',
            'min_speakers',
            'max_speakers',
        ):
            if args[param] is not None:
                raise pxt.Error(f'`{param}` can only be set if `diarize=True`')

    device = resolve_torch_device('auto', allow_mps=False)
    compute_type = compute_type or ('float16' if device == 'cuda' else 'int8')
    transcription_model = _lookup_transcription_model(model, device, compute_type)
    audio_array: np.ndarray = whisperx.load_audio(audio)
    kwargs: dict[str, Any] = {'language': language, 'task': task}
    if chunk_size is not None:
        kwargs['chunk_size'] = chunk_size
    result: dict[str, Any] = transcription_model.transcribe(audio_array, batch_size=16, **kwargs)

    if diarize:
        # Alignment
        alignment_model, metadata = _lookup_alignment_model(result['language'], device, alignment_model_name)
        kwargs = {}
        if interpolate_method is not None:
            kwargs['interpolate_method'] = interpolate_method
        if return_char_alignments is not None:
            kwargs['return_char_alignments'] = return_char_alignments
        result = whisperx.align(result['segments'], alignment_model, metadata, audio_array, device, **kwargs)

        # Diarization
        diarization_model = _lookup_diarization_model(device, diarization_model_name)
        diarization_segments = diarization_model(
            audio_array, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers
        )
        result = whisperx.assign_word_speakers(diarization_segments, result)

    return result


def _lookup_transcription_model(model: str, device: str, compute_type: str) -> 'FasterWhisperPipeline':
    import whisperx

    key = (model, device, compute_type)
    if key not in _model_cache:
        transcription_model = whisperx.load_model(model, device, compute_type=compute_type)
        _model_cache[key] = transcription_model
    return _model_cache[key]


def _lookup_alignment_model(language_code: str, device: str, model_name: str | None) -> tuple['Wav2Vec2Model', dict]:
    import whisperx

    key = (language_code, device, model_name)
    if key not in _alignment_model_cache:
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device, model_name=model_name)
        _alignment_model_cache[key] = (model, metadata)
    return _alignment_model_cache[key]


def _lookup_diarization_model(device: str, model_name: str | None) -> 'DiarizationPipeline':
    from whisperx.diarize import DiarizationPipeline

    key = (device, model_name)
    if key not in _diarization_model_cache:
        auth_token = Config.get().get_string_value('auth_token', section='hf')
        kwargs: dict[str, Any] = {'device': device, 'use_auth_token': auth_token}
        if model_name is not None:
            kwargs['model_name'] = model_name
        _diarization_model_cache[key] = DiarizationPipeline(**kwargs)
    return _diarization_model_cache[key]


_model_cache: dict[tuple[str, str, str], 'FasterWhisperPipeline'] = {}
_alignment_model_cache: dict[tuple[str, str, str | None], tuple['Wav2Vec2Model', dict]] = {}
_diarization_model_cache: dict[tuple[str, str | None], 'DiarizationPipeline'] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

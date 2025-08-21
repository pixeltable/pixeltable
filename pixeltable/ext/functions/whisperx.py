from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

from pixeltable.config import Config
from pixeltable.functions.util import resolve_torch_device
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from transformers import Wav2Vec2Model
    from whisperx.asr import FasterWhisperPipeline  # type: ignore[import-untyped]
    from whisperx.diarize import DiarizationPipeline  # type: ignore[import-untyped]

import pixeltable as pxt


@pxt.udf
def transcribe(
    audio: pxt.Audio,
    *,
    model: str,
    diarize: bool = False,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    task: Optional[Literal['transcribe', 'translate']] = None,
    chunk_size: Optional[int] = None,
    alignment_model_name: Optional[str] = None,
    interpolate_method: Optional[str] = None,
    return_char_alignments: Optional[bool] = None,
    diarization_model_name: Optional[str] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
) -> dict:
    """
    Transcribe an audio file using WhisperX.

    This UDF runs a transcription model _locally_ using the WhisperX library,
    equivalent to the WhisperX `transcribe` function, as described in the
    [WhisperX library documentation](https://github.com/m-bain/whisperX).

    WhisperX is part of the `pixeltable.ext` package: long-term support in Pixeltable is not guaranteed.

    __Requirements:__

    - `pip install whisperx`

    Args:
        audio: The audio file to transcribe.
        model: The name of the model to use for transcription.

    See the [WhisperX library documentation](https://github.com/m-bain/whisperX) for details
    on the remaining parameters.

    Returns:
        A dictionary containing the transcription and various other metadata.

    Examples:
        Add a computed column that applies the model `tiny.en` to an existing Pixeltable column `tbl.audio`
        of the table `tbl`:

        >>> tbl.add_computed_column(result=transcribe(tbl.audio, model='tiny.en'))
    """
    import whisperx  # type: ignore[import-untyped]

    if not diarize:
        if alignment_model_name is not None:
            raise pxt.Error('`alignment_model_name` can only be set if `diarize=True`')
        if interpolate_method is not None:
            raise pxt.Error('`interpolate_method` can only be set if `diarize=True`')
        if return_char_alignments is not None:
            raise pxt.Error('`return_char_alignments` can only be set if `diarize=True`')
        if diarization_model_name is not None:
            raise pxt.Error('`diarization_model_name` can only be set if `diarize=True`')
        if num_speakers is not None:
            raise pxt.Error('`num_speakers` can only be set if `diarize=True`')
        if min_speakers is not None:
            raise pxt.Error('`min_speakers` can only be set if `diarize=True`')
        if max_speakers is not None:
            raise pxt.Error('`max_speakers` can only be set if `diarize=True`')

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


def _lookup_alignment_model(language_code: str, device: str, model_name: Optional[str]) -> tuple['Wav2Vec2Model', dict]:
    import whisperx

    key = (language_code, device, model_name)
    if key not in _alignment_model_cache:
        model, metadata = whisperx.load_align_model(language_code=language_code, device=device, model_name=model_name)
        _alignment_model_cache[key] = (model, metadata)
    return _alignment_model_cache[key]


def _lookup_diarization_model(device: str, model_name: Optional[str]) -> 'DiarizationPipeline':
    from whisperx.diarize import DiarizationPipeline

    key = (device, model_name)
    if key not in _diarization_model_cache:
        auth_token = Config.get().get_string_value('auth_token', section='hugging_face')
        kwargs: dict[str, Any] = {'device': device, 'use_auth_token': auth_token}
        if model_name is not None:
            kwargs['model_name'] = model_name
        _diarization_model_cache[key] = DiarizationPipeline(**kwargs)
    return _diarization_model_cache[key]


_model_cache: dict[tuple[str, str, str], 'FasterWhisperPipeline'] = {}
_alignment_model_cache: dict[tuple[str, str, Optional[str]], tuple['Wav2Vec2Model', dict]] = {}
_diarization_model_cache: dict[tuple[str, Optional[str]], 'DiarizationPipeline'] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

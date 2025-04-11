from typing import TYPE_CHECKING, Optional

from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from whisperx.asr import FasterWhisperPipeline  # type: ignore[import-untyped]

import pixeltable as pxt


@pxt.udf
def transcribe(
    audio: pxt.Audio,
    *,
    model: str,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    chunk_size: int = 30,
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
    import torch
    import whisperx  # type: ignore[import-untyped]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_type = compute_type or ('float16' if device == 'cuda' else 'int8')
    model = _lookup_model(model, device, compute_type)
    audio_array = whisperx.load_audio(audio)
    result = model.transcribe(audio_array, batch_size=16, language=language, chunk_size=chunk_size)
    return result


def _lookup_model(model_id: str, device: str, compute_type: str) -> 'FasterWhisperPipeline':
    import whisperx

    key = (model_id, device, compute_type)
    if key not in _model_cache:
        model = whisperx.load_model(model_id, device, compute_type=compute_type)
        _model_cache[key] = model
    return _model_cache[key]


_model_cache: dict[tuple[str, str, str], 'FasterWhisperPipeline'] = {}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

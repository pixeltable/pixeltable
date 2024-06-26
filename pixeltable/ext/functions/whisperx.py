from typing import Optional, TYPE_CHECKING

import torch

from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from whisperx.asr import FasterWhisperPipeline

import pixeltable as pxt


@pxt.udf(param_types=[pxt.AudioType(), pxt.StringType(), pxt.StringType(), pxt.StringType(), pxt.IntType()])
def transcribe(
    audio: str, *, model: str, compute_type: Optional[str] = None, language: Optional[str] = None, chunk_size: int = 30
) -> dict:
    import whisperx

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


_model_cache = {}


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

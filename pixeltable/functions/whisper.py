from typing import TYPE_CHECKING, Optional

import pixeltable as pxt

if TYPE_CHECKING:
    from whisper import Whisper


@pxt.udf(param_types=[pxt.AudioType(), pxt.StringType(), pxt.JsonType(nullable=True), pxt.FloatType(nullable=True),
                      pxt.FloatType(nullable=True), pxt.FloatType(nullable=True),
                      pxt.BoolType(), pxt.StringType(nullable=True), pxt.BoolType(), pxt.StringType(),
                      pxt.StringType(), pxt.StringType(), pxt.FloatType(nullable=True), pxt.JsonType(nullable=True)])
def transcribe(
        audio: str,
        *,
        model: str,
        temperature: Optional[list[float]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        clip_timestamps: str = "0",
        hallucination_silence_threshold: Optional[float] = None,
        decode_options: Optional[dict] = None
) -> dict:
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
        clip_timestamps=clip_timestamps,
        hallucination_silence_threshold=hallucination_silence_threshold,
        **decode_options
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

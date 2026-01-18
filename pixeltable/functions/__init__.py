"""
General Pixeltable UDFs.

This parent module contains general-purpose UDFs that apply to multiple data types.
"""

# ruff: noqa: F401

from pixeltable.utils.code import local_public_names

from . import (
    anthropic,
    audio,
    bedrock,
    date,
    deepseek,
    fal,
    fireworks,
    gemini,
    groq,
    huggingface,
    image,
    jina,
    json,
    llama_cpp,
    math,
    mistralai,
    net,
    ollama,
    openai,
    openrouter,
    replicate,
    reve,
    string,
    timestamp,
    together,
    twelvelabs,
    uuid,
    video,
    vision,
    voyageai,
    whisper,
    whisperx,
    yolox,
)
from .globals import count, map, max, mean, min, sum

__all__ = local_public_names(__name__, exclude=['globals']) + local_public_names(globals.__name__)


def __dir__() -> list[str]:
    return __all__

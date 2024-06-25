import base64
from typing import Optional, TYPE_CHECKING

import PIL.Image
import numpy as np

import io
import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import together


@env.register_client('together')
def _(api_key: str) -> 'together.Together':
    import together

    return together.Together(api_key=api_key)


def _together_client() -> 'together.Together':
    return env.Env.get().get_client('together')


@pxt.udf
def completions(
    prompt: str,
    *,
    model: str,
    max_tokens: Optional[int] = None,
    stop: Optional[list] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    logprobs: Optional[int] = None,
    echo: Optional[bool] = None,
    n: Optional[int] = None,
    safety_model: Optional[str] = None,
) -> dict:
    return (
        _together_client()
        .completions.create(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            logprobs=logprobs,
            echo=echo,
            n=n,
            safety_model=safety_model,
        )
        .dict()
    )


@pxt.udf
def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
    logprobs: Optional[int] = None,
    echo: Optional[bool] = None,
    n: Optional[int] = None,
    safety_model: Optional[str] = None,
    response_format: Optional[dict] = None,
    tools: Optional[dict] = None,
    tool_choice: Optional[dict] = None,
) -> dict:
    return (
        _together_client()
        .chat.completions.create(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            logprobs=logprobs,
            echo=echo,
            n=n,
            safety_model=safety_model,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )
        .dict()
    )


_embedding_dimensions_cache = {
    'togethercomputer/m2-bert-80M-2k-retrieval': 768,
    'togethercomputer/m2-bert-80M-8k-retrieval': 768,
    'togethercomputer/m2-bert-80M-32k-retrieval': 768,
    'WhereIsAI/UAE-Large-V1': 1024,
    'BAAI/bge-large-en-v1.5': 1024,
    'BAAI/bge-base-en-v1.5': 768,
    'sentence-transformers/msmarco-bert-base-dot-v5': 768,
    'bert-base-uncased': 768,
}


@pxt.udf(batch_size=32, return_type=pxt.ArrayType((None,), dtype=pxt.FloatType()))
def embeddings(input: Batch[str], *, model: str) -> Batch[np.ndarray]:
    result = _together_client().embeddings.create(input=input, model=model)
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str) -> pxt.ArrayType:
    if model not in _embedding_dimensions_cache:
        # TODO: find some other way to retrieve a sample
        return pxt.ArrayType((None,), dtype=pxt.FloatType())
    dimensions = _embedding_dimensions_cache[model]
    return pxt.ArrayType((dimensions,), dtype=pxt.FloatType())


@pxt.udf
def image_generations(
    prompt: str,
    *,
    model: str,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    negative_prompt: Optional[str] = None,
) -> PIL.Image.Image:
    # TODO(aaron-siegel): Decompose CPU/GPU ops into separate functions
    result = _together_client().images.generate(
        prompt=prompt, model=model, steps=steps, seed=seed, height=height, width=width, negative_prompt=negative_prompt
    )
    b64_str = result.data[0].b64_json
    b64_bytes = base64.b64decode(b64_str)
    img = PIL.Image.open(io.BytesIO(b64_bytes))
    img.load()
    return img


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

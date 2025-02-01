"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Together AI API. In order to use them, you must
first `pip install together` and configure your Together AI credentials, as described in
the [Working with Together AI](https://pixeltable.readme.io/docs/together-ai) tutorial.
"""

import base64
import io
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import numpy as np
import PIL.Image
import requests
import tenacity

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import together  # type: ignore[import-untyped]


@env.register_client('together')
def _(api_key: str) -> 'together.Together':
    import together

    return together.Together(api_key=api_key)


def _together_client() -> 'together.Together':
    return env.Env.get().get_client('together')


T = TypeVar('T')


def _retry(fn: Callable[..., T]) -> Callable[..., T]:
    import together

    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(together.error.RateLimitError),
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(20),
    )(fn)


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
    """
    Generate completions based on a given prompt using a specified model.

    Equivalent to the Together AI `completions` API endpoint.
    For additional details, see: [https://docs.together.ai/reference/completions-1](https://docs.together.ai/reference/completions-1)

    __Requirements:__

    - `pip install together`

    Args:
        prompt: A string providing context for the model to complete.
        model: The name of the model to query.

    For details on the other parameters, see: [https://docs.together.ai/reference/completions-1](https://docs.together.ai/reference/completions-1)

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistralai/Mixtral-8x7B-v0.1` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> tbl['response'] = completions(tbl.prompt, model='mistralai/Mixtral-8x7B-v0.1')
    """
    return _retry(_together_client().completions.create)(
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
    ).dict()


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
    """
    Generate chat completions based on a given prompt using a specified model.

    Equivalent to the Together AI `chat/completions` API endpoint.
    For additional details, see: [https://docs.together.ai/reference/chat-completions-1](https://docs.together.ai/reference/chat-completions-1)

    __Requirements:__

    - `pip install together`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The name of the model to query.

    For details on the other parameters, see: [https://docs.together.ai/reference/chat-completions-1](https://docs.together.ai/reference/chat-completions-1)

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistralai/Mixtral-8x7B-v0.1` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl['response'] = chat_completions(messages, model='mistralai/Mixtral-8x7B-v0.1')
    """
    return _retry(_together_client().chat.completions.create)(
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
    ).dict()


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


@pxt.udf(batch_size=32)
def embeddings(input: Batch[str], *, model: str) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Query an embedding model for a given string of text.

    Equivalent to the Together AI `embeddings` API endpoint.
    For additional details, see: [https://docs.together.ai/reference/embeddings-2](https://docs.together.ai/reference/embeddings-2)

    __Requirements:__

    - `pip install together`

    Args:
        input: A string providing the text for the model to embed.
        model: The name of the embedding model to use.

    Returns:
        An array representing the application of the given embedding to `input`.

    Examples:
        Add a computed column that applies the model `togethercomputer/m2-bert-80M-8k-retrieval`
        to an existing Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl['response'] = embeddings(tbl.text, model='togethercomputer/m2-bert-80M-8k-retrieval')
    """
    result = _retry(_together_client().embeddings.create)(input=input, model=model)
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
    """
    Generate images based on a given prompt using a specified model.

    Equivalent to the Together AI `images/generations` API endpoint.
    For additional details, see: [https://docs.together.ai/reference/post_images-generations](https://docs.together.ai/reference/post_images-generations)

    __Requirements:__

    - `pip install together`

    Args:
        prompt: A description of the desired images.
        model: The model to use for image generation.

    For details on the other parameters, see: [https://docs.together.ai/reference/post_images-generations](https://docs.together.ai/reference/post_images-generations)

    Returns:
        The generated image.

    Examples:
        Add a computed column that applies the model `stabilityai/stable-diffusion-xl-base-1.0`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl['response'] = image_generations(tbl.prompt, model='stabilityai/stable-diffusion-xl-base-1.0')
    """
    result = _retry(_together_client().images.generate)(
        prompt=prompt, model=model, steps=steps, seed=seed, height=height, width=width, negative_prompt=negative_prompt
    )
    if result.data[0].b64_json is not None:
        b64_bytes = base64.b64decode(result.data[0].b64_json)
        img = PIL.Image.open(io.BytesIO(b64_bytes))
        img.load()
        return img
    if result.data[0].url is not None:
        try:
            resp = requests.get(result.data[0].url)
            with io.BytesIO(resp.content) as fp:
                image = PIL.Image.open(fp)
                image.load()
                return image
        except Exception as exc:
            raise excs.Error('Failed to download generated image from together.ai.') from exc
    raise excs.Error('Response does not contain a generated image.')


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

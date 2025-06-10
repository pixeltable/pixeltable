"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Together AI API. In order to use them, you must
first `pip install together` and configure your Together AI credentials, as described in
the [Working with Together AI](https://pixeltable.readme.io/docs/together-ai) tutorial.
"""

import base64
import io
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

import numpy as np
import PIL.Image
import requests
import tenacity

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import together  # type: ignore[import-untyped]


@env.register_client('together')
def _(api_key: str) -> 'together.AsyncTogether':
    import together

    return together.AsyncTogether(api_key=api_key)


def _together_client() -> 'together.AsyncTogether':
    return env.Env.get().get_client('together')


T = TypeVar('T')


def _retry(fn: Callable[..., T]) -> Callable[..., T]:
    import together

    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(together.error.RateLimitError),
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(20),
    )(fn)


@pxt.udf(resource_pool='request-rate:together:chat')
async def completions(prompt: str, *, model: str, model_kwargs: Optional[dict[str, Any]] = None) -> dict:
    """
    Generate completions based on a given prompt using a specified model.

    Equivalent to the Together AI `completions` API endpoint.
    For additional details, see: <https://docs.together.ai/reference/completions-1>

    Request throttling:
    Applies the rate limit set in the config (section `together.rate_limits`, key `chat`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install together`

    Args:
        prompt: A string providing context for the model to complete.
        model: The name of the model to query.
        model_kwargs: Additional keyword arguments for the Together `completions` API.
            For details on the available parameters, see: <https://docs.together.ai/reference/completions-1>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistralai/Mixtral-8x7B-v0.1` to an existing Pixeltable column
        `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=completions(tbl.prompt, model='mistralai/Mixtral-8x7B-v0.1'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    result = await _together_client().completions.create(prompt=prompt, model=model, **model_kwargs)
    return result.dict()


@pxt.udf(resource_pool='request-rate:together:chat')
async def chat_completions(
    messages: list[dict[str, str]], *, model: str, model_kwargs: Optional[dict[str, Any]] = None
) -> dict:
    """
    Generate chat completions based on a given prompt using a specified model.

    Equivalent to the Together AI `chat/completions` API endpoint.
    For additional details, see: <https://docs.together.ai/reference/chat-completions-1>

    Request throttling:
    Applies the rate limit set in the config (section `together.rate_limits`, key `chat`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install together`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The name of the model to query.
        model_kwargs: Additional keyword arguments for the Together `chat/completions` API.
            For details on the available parameters, see: <https://docs.together.ai/reference/chat-completions-1>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistralai/Mixtral-8x7B-v0.1` to an existing Pixeltable column
        `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(response=chat_completions(messages, model='mistralai/Mixtral-8x7B-v0.1'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    result = await _together_client().chat.completions.create(messages=messages, model=model, **model_kwargs)
    return result.dict()


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


@pxt.udf(batch_size=32, resource_pool='request-rate:together:embeddings')
async def embeddings(input: Batch[str], *, model: str) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Query an embedding model for a given string of text.

    Equivalent to the Together AI `embeddings` API endpoint.
    For additional details, see: <https://docs.together.ai/reference/embeddings-2>

    Request throttling:
    Applies the rate limit set in the config (section `together.rate_limits`, key `embeddings`). If no rate
    limit is configured, uses a default of 600 RPM.

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

        >>> tbl.add_computed_column(response=embeddings(tbl.text, model='togethercomputer/m2-bert-80M-8k-retrieval'))
    """
    result = await _together_client().embeddings.create(input=input, model=model)
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str) -> ts.ArrayType:
    if model not in _embedding_dimensions_cache:
        # TODO: find some other way to retrieve a sample
        return ts.ArrayType((None,), dtype=ts.FloatType())
    dimensions = _embedding_dimensions_cache[model]
    return ts.ArrayType((dimensions,), dtype=ts.FloatType())


@pxt.udf(resource_pool='request-rate:together:images')
async def image_generations(
    prompt: str, *, model: str, model_kwargs: Optional[dict[str, Any]] = None
) -> PIL.Image.Image:
    """
    Generate images based on a given prompt using a specified model.

    Equivalent to the Together AI `images/generations` API endpoint.
    For additional details, see: <https://docs.together.ai/reference/post_images-generations>

    Request throttling:
    Applies the rate limit set in the config (section `together.rate_limits`, key `images`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install together`

    Args:
        prompt: A description of the desired images.
        model: The model to use for image generation.
        model_kwargs: Additional keyword args for the Together `images/generations` API.
            For details on the available parameters, see: <https://docs.together.ai/reference/post_images-generations>

    Returns:
        The generated image.

    Examples:
        Add a computed column that applies the model `stabilityai/stable-diffusion-xl-base-1.0`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=image_generations(tbl.prompt, model='stabilityai/stable-diffusion-xl-base-1.0')
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    result = await _together_client().images.generate(prompt=prompt, model=model, **model_kwargs)
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


def __dir__() -> list[str]:
    return __all__

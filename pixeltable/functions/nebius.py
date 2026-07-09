"""
Pixeltable UDFs for Nebius Token Factory models.

Provides integration with Nebius Token Factory's language and embedding models. In order to use
them, you must first `pip install openai` and configure your Nebius Token Factory API key.
"""

import json
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import openai


@env.register_client('nebius', credential_param='api_key')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.tokenfactory.nebius.com/v1/',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _nebius_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('nebius')


@pxt.udf(is_deterministic=False, resource_pool='request-rate:nebius')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the Nebius Token Factory `chat/completions` API endpoint.
    For additional details, see: <https://docs.nebius.com/studio/inference/api-reference>

    Nebius Token Factory exposes an OpenAI-compatible API, so you will need to install the
    `openai` package to use this UDF.

    Request throttling:
    Applies the rate limit set in the config (section `nebius`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the Nebius API documentation.
        model: The model to use for chat completion.
        model_kwargs: Additional keyword args for the Nebius `chat/completions` API.
            For details on the available parameters, see: <https://docs.nebius.com/studio/inference/api-reference>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `meta-llama/Llama-3.3-70B-Instruct`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt},
        ... ]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages, model='meta-llama/Llama-3.3-70B-Instruct'
        ...     )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    if tools is not None:
        model_kwargs['tools'] = [{'type': 'function', 'function': tool} for tool in tools]

    if tool_choice is not None:
        if tool_choice['auto']:
            model_kwargs['tool_choice'] = 'auto'
        elif tool_choice['required']:
            model_kwargs['tool_choice'] = 'required'
        else:
            assert tool_choice['tool'] is not None
            model_kwargs['tool_choice'] = {'type': 'function', 'function': {'name': tool_choice['tool']}}

    if tool_choice is not None and not tool_choice['parallel_tool_calls']:
        if 'extra_body' not in model_kwargs:
            model_kwargs['extra_body'] = {}
        model_kwargs['extra_body']['parallel_tool_calls'] = False

    result = await _nebius_client().chat.completions.with_raw_response.create(
        messages=messages, model=model, **model_kwargs
    )

    return json.loads(result.text)


_embedding_dimensions_cache: dict[str, int] = {'Qwen/Qwen3-Embedding-8B': 4096}


@pxt.udf(batch_size=32, resource_pool='request-rate:nebius')
async def embeddings(
    input: Batch[str], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Query an embedding model for a given string of text.

    Equivalent to the Nebius Token Factory `embeddings` API endpoint.
    For additional details, see: <https://docs.nebius.com/studio/inference/api-reference>

    Request throttling:
    Applies the rate limit set in the config (section `nebius`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        input: A string providing the text for the model to embed.
        model: The name of the embedding model to use.
        model_kwargs: Additional keyword args for the Nebius `embeddings` API, e.g. `dimensions` to
            request a truncated embedding for models that support it (see the note below).

    Returns:
        An array representing the application of the given embedding to `input`.

    Examples:
        Add a computed column that applies the model `Qwen/Qwen3-Embedding-8B`
        to an existing Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=embeddings(tbl.text, model='Qwen/Qwen3-Embedding-8B')
        ... )

        `Qwen/Qwen3-Embedding-8B` produces 4096-dimensional embeddings by default, which exceed
        pgvector's dimensionality limits for `add_embedding_index()` (2000 for `fp32` precision,
        4000 for the default `fp16`). Request a smaller, indexable size via `model_kwargs`:

        >>> tbl.add_embedding_index(
        ...     'text',
        ...     embedding=embeddings.using(
        ...         model='Qwen/Qwen3-Embedding-8B', model_kwargs={'dimensions': 1024}
        ...     ),
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}
    result = await _nebius_client().embeddings.create(input=input, model=model, **model_kwargs)
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str, model_kwargs: dict[str, Any] | None = None) -> ts.ArrayType:
    dimensions = model_kwargs.get('dimensions') if model_kwargs is not None else None
    if dimensions is None:
        dimensions = _embedding_dimensions_cache.get(model)  # `None` if unknown model
    return ts.ArrayType((dimensions,), dtype=ts.FloatType())


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

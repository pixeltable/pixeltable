"""
Pixeltable UDFs for Cheaper Inference models.

Provides integration with the language and embedding models served by Cheaper Inference, an OpenAI-compatible
LLM gateway. In order to use them, you must first `pip install openai` and configure your Cheaper Inference API key,
by setting the `CHEAPERINFERENCE_API_KEY` environment variable or `cheaperinference.api_key` in the Pixeltable
config file.
"""

import copy
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import env, exceptions as excs
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import openai


@env.register_client('cheaperinference', credential_param='api_key')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(api_key=api_key, base_url='https://api.cheaperinference.com/v1')


def _cheaperinference_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('cheaperinference')


@pxt.udf(is_deterministic=False, resource_pool='request-rate:cheaperinference')
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

    Equivalent to the Cheaper Inference `chat/completions` API endpoint.
    For additional details, see: <https://cheaperinference.com/docs>

    Cheaper Inference exposes an OpenAI-compatible API, so you will need to install the
    `openai` package to use this UDF.

    Request throttling:
    Applies the rate limit set in the config (section `cheaperinference`, key `rate_limit`). If no rate
    limit is configured, a default rate limit is applied.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the Cheaper Inference API
            documentation.
        model: The model to use for chat completion.
        model_kwargs: Additional keyword args for the Cheaper Inference `chat/completions` API.
            For details on the available parameters, see: <https://cheaperinference.com/docs>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gpt-5.4-mini`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt},
        ... ]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(messages, model='gpt-5.4-mini')
        ... )

        You can also include images in the messages list, for vision-capable models such as
        `gpt-5.4-mini`, by passing image data directly in the input dictionary, in
        the `'image_url'` field of the message content, as in this example:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'type': 'text', 'text': "What's in this image?"},
        ...             {'type': 'image_url', 'image_url': tbl.image},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(messages, model='gpt-5.4-mini')
        ... )
    """
    env.Env.get().require_package('openai')

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

    # Serialize any images in `messages`
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get('type') == 'image_url'
                    and isinstance(part.get('image_url'), PIL.Image.Image)
                ):
                    b64_encoded_image = to_base64(part['image_url'], format='png')
                    part['image_url'] = {'url': f'data:image/png;base64,{b64_encoded_image}'}

    result = await _cheaperinference_client().chat.completions.with_raw_response.create(
        messages=messages, model=model, **model_kwargs
    )

    result_json = json.loads(result.text)
    if 'error' in result_json:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, str(result_json['error']), provider='cheaperinference'
        )
    return result_json


_embedding_dimensions: dict[str, int] = {'text-embedding-3-small': 1536}


@pxt.udf(batch_size=32, resource_pool='request-rate:cheaperinference')
async def embeddings(
    input: Batch[str], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Query an embedding model for a given string of text.

    Equivalent to the Cheaper Inference `embeddings` API endpoint.
    For additional details, see: <https://cheaperinference.com/docs>

    Request throttling:
    Applies the rate limit set in the config (section `cheaperinference`, key `rate_limit`). If no rate
    limit is configured, a default rate limit is applied.

    __Requirements:__

    - `pip install openai`

    Args:
        input: A string providing the text for the model to embed.
        model: The name of the embedding model to use.
        model_kwargs: Additional keyword args for the Cheaper Inference `embeddings` API.

    Returns:
        An array representing the application of the given embedding to `input`.

    Examples:
        Add a computed column that applies the model `text-embedding-3-small`
        to an existing Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl.add_computed_column(
        ...     response=embeddings(tbl.text, model='text-embedding-3-small')
        ... )

        Add an embedding index to an existing column `text`, using the model `text-embedding-3-small`:

        >>> tbl.add_embedding_index(
        ...     'text', embedding=embeddings.using(model='text-embedding-3-small')
        ... )
    """
    env.Env.get().require_package('openai')

    if model_kwargs is None:
        model_kwargs = {}

    result = await _cheaperinference_client().embeddings.with_raw_response.create(
        input=input, model=model, encoding_format='float', **model_kwargs
    )

    result_json = json.loads(result.text)
    if 'error' in result_json:
        raise excs.ExternalServiceError(
            excs.ErrorCode.PROVIDER_ERROR, str(result_json['error']), provider='cheaperinference'
        )
    return [np.array(data['embedding'], dtype=np.float64) for data in result_json['data']]


@embeddings.conditional_return_type
def _(model: str, model_kwargs: dict[str, Any] | None = None) -> ts.ArrayType:
    dimensions = model_kwargs.get('dimensions') if model_kwargs is not None else None
    if dimensions is None:
        dimensions = _embedding_dimensions.get(model)  # `None` if unknown model
    return ts.ArrayType((dimensions,), dtype=ts.FloatType())


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

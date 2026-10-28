"""
Pixeltable UDFs for OpenLLM.

OpenLLM connects your AI subscriptions, provider API keys, and local models, and serves them through one
OpenAI-compatible endpoint. By default, these UDFs call the local OpenLLM daemon at `http://127.0.0.1:8787/v1`,
which needs no configuration. Set `OPENLLM_BASE_URL` to use another endpoint, and `OPENLLM_API_KEY` if that
endpoint requires authentication. To get started, see the [OpenLLM quickstart](https://docs.openllm.sh/quickstart).
"""

import copy
import json
from typing import TYPE_CHECKING, Any

import PIL.Image

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

if TYPE_CHECKING:
    import openai


# The local daemon accepts requests without an `Authorization` header but rejects unknown keys. The OpenAI SDK
# requires a non-empty key, so a placeholder is used and the header is omitted from each request.
_NO_API_KEY = 'openllm-no-key'


@register_client('openllm', credential_param='api_key')
def _(api_key: str = _NO_API_KEY, base_url: str = 'http://127.0.0.1:8787/v1') -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)


def _openllm_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('openllm')


def _omit_placeholder_auth(client: 'openai.AsyncOpenAI', model_kwargs: dict[str, Any]) -> None:
    import openai

    if client.api_key == _NO_API_KEY:
        model_kwargs['extra_headers'] = {'Authorization': openai.Omit(), **model_kwargs.get('extra_headers', {})}


@pxt.udf(is_deterministic=False, resource_pool='request-rate:openllm')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given chat conversation through OpenLLM.

    Equivalent to the OpenAI-compatible `chat/completions` endpoint of OpenLLM.
    For additional details, see: <https://docs.openllm.sh>

    The model can be a fallback alias such as `plus`, which moves to the next model in its chain when a provider
    is rate-limited or unavailable, or any model ID listed by the OpenLLM `models` endpoint.

    Request throttling:
    Applies the rate limit set in the config (section `openllm`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, in the OpenAI chat format.
        model: The model or fallback alias to use for chat completion.
        model_kwargs: Additional OpenAI-compatible parameters for the `chat/completions` endpoint.
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the `plus` alias to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        >>> tbl.add_computed_column(response=chat_completions(messages, model='plus'))

        You can also include images in the messages list, for vision-capable models, by passing image data
        directly in the `'image_url'` field of the message content, as in this example:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'type': 'text', 'text': "What's in this image?"},
        ...             {'type': 'image_url', 'image_url': tbl.image},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(response=chat_completions(messages, model='plus'))
    """
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    Env.get().require_package('openai')

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
        model_kwargs['extra_body'] = {**model_kwargs.get('extra_body', {}), 'parallel_tool_calls': False}

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

    client = _openllm_client()
    _omit_placeholder_auth(client, model_kwargs)
    result = await client.chat.completions.with_raw_response.create(messages=messages, model=model, **model_kwargs)

    return json.loads(result.text)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:openllm')
async def responses(
    input: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given input through OpenLLM.

    Equivalent to the OpenAI-compatible `responses` endpoint of OpenLLM.
    For additional details, see: <https://docs.openllm.sh>

    The model can be a fallback alias such as `plus`, which moves to the next model in its chain when a provider
    is rate-limited or unavailable, or any model ID listed by the OpenLLM `models` endpoint.

    Request throttling:
    Applies the rate limit set in the config (section `openllm`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        input: A list of input items for the model, in the OpenAI Responses format.
        model: The model or fallback alias to use for generating a response.
        model_kwargs: Additional OpenAI-compatible parameters for the `responses` endpoint, such as
            `instructions`, `temperature` or `max_output_tokens`.
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata. The response text is in the `output` list.

    Examples:
        Add a computed column that applies the `plus` alias to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        >>> tbl.add_computed_column(response=responses(messages, model='plus'))

        You can also include images in the input list, for vision-capable models, by passing image data
        directly with the `input_image` content type:

        >>> messages = [
        ...     {
        ...         'role': 'user',
        ...         'content': [
        ...             {'type': 'input_text', 'text': "What's in this image?"},
        ...             {'type': 'input_image', 'image_url': tbl.image},
        ...         ],
        ...     }
        ... ]
        >>> tbl.add_computed_column(response=responses(messages, model='plus'))
    """
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

    Env.get().require_package('openai')

    # The Responses API uses a flat function tool format
    if tools is not None:
        model_kwargs['tools'] = [{'type': 'function', **tool} for tool in tools]

    if tool_choice is not None:
        if tool_choice['auto']:
            model_kwargs['tool_choice'] = 'auto'
        elif tool_choice['required']:
            model_kwargs['tool_choice'] = 'required'
        else:
            assert tool_choice['tool'] is not None
            model_kwargs['tool_choice'] = {'type': 'function', 'name': tool_choice['tool']}

    if tool_choice is not None and not tool_choice['parallel_tool_calls']:
        model_kwargs['parallel_tool_calls'] = False

    # Serialize any images in `input`
    input = copy.deepcopy(input)
    for item in input:
        content = item.get('content') if isinstance(item, dict) else None
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get('type') == 'input_image'
                    and isinstance(part.get('image_url'), PIL.Image.Image)
                ):
                    b64_encoded_image = to_base64(part['image_url'], format='png')
                    part['image_url'] = f'data:image/png;base64,{b64_encoded_image}'

    client = _openllm_client()
    _omit_placeholder_auth(client, model_kwargs)
    result = await client.responses.with_raw_response.create(input=input, model=model, **model_kwargs)

    return json.loads(result.text)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

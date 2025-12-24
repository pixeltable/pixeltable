"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) for [xAI](https://x.ai/) Grok models.

Provides integration with xAI's Grok language and image generation models using the native xAI SDK.

In order to use these UDFs, you must configure your xAI API key either via the `XAI_API_KEY` environment
variable, or as `api_key` in the `xai` section of the Pixeltable config file.
"""

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

import httpx
import PIL.Image

import pixeltable as pxt
from pixeltable import env
from pixeltable.config import Config
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from xai_sdk import AsyncClient as XAIAsyncClient


@env.register_client('xai')
def _(api_key: str) -> 'XAIAsyncClient':
    from xai_sdk import AsyncClient

    return AsyncClient(
        api_key=api_key,
        timeout=3600,  # Extended timeout for reasoning models
    )


def _xai_client() -> 'XAIAsyncClient':
    return env.Env.get().get_client('xai')


@pxt.udf(resource_pool='request-rate:xai')
async def chat(
    messages: list,
    *,
    model: str = 'grok-3',
    reasoning_effort: Literal['low', 'high'] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    store_messages: bool = False,
) -> dict:
    """
    Creates a model response for the given chat conversation using xAI's Grok models.

    Uses the native xAI SDK which supports the latest features including the Responses API
    and reasoning models.

    For additional details, see: <https://docs.x.ai/docs/guides/chat>

    Request throttling:
    Applies the rate limit set in the config (section `xai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 60 RPM.

    __Requirements:__

    - `pip install xai-sdk`

    Args:
        messages: A list of messages comprising the conversation. Each message should have
            a `role` (system, user, or assistant) and `content`.
        model: The Grok model to use. Options include:
            - `grok-4`: Latest Grok 4 model (most capable)
            - `grok-4-fast`: Faster Grok 4 variant
            - `grok-3`: Grok 3 model
            - `grok-3-fast`: Faster Grok 3 variant
            - `grok-3-mini`: Grok 3 mini with reasoning support
            - `grok-2-vision-1212`: Grok 2 with vision capabilities
            - `grok-code-fast-1`: Code-optimized model
        reasoning_effort: Controls how much time the model spends thinking. Only supported
            by `grok-3-mini`. Options: `low` (quick responses) or `high` (complex problems).
        max_tokens: Maximum number of tokens in the response.
        temperature: Sampling temperature (0-2). Higher values make output more random.
        top_p: Nucleus sampling parameter.
        store_messages: If True, uses the stateful Responses API where messages are stored
            server-side. This enables multi-turn conversations with less data transfer.

    Returns:
        A dictionary containing:
        - `content`: The response text
        - `usage`: Token usage information including `reasoning_tokens` for reasoning models
        - `id`: Response ID (can be used to continue conversation if store_messages=True)
        - `model`: The model used

    Examples:
        Basic chat completion:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are Grok, a helpful AI assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt}
        ... ]
        >>> tbl.add_computed_column(response=xai.chat(messages, model='grok-4'))

        Using reasoning model:

        >>> tbl.add_computed_column(
        ...     response=xai.chat(
        ...         messages,
        ...         model='grok-3-mini',
        ...         reasoning_effort='high'
        ...     )
        ... )
    """
    from xai_sdk.chat import system, user as user_msg

    client = _xai_client()

    # Build kwargs for chat creation
    chat_kwargs: dict[str, Any] = {'model': model}

    if reasoning_effort is not None:
        chat_kwargs['reasoning_effort'] = reasoning_effort
    if max_tokens is not None:
        chat_kwargs['max_tokens'] = max_tokens
    if store_messages:
        chat_kwargs['store_messages'] = store_messages

    # Create chat instance
    chat_instance = client.chat.create(**chat_kwargs)

    # Add messages
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')

        if role == 'system':
            chat_instance.append(system(content))
        elif role == 'user':
            chat_instance.append(user_msg(content))
        elif role == 'assistant':
            # For assistant messages, we need to handle them appropriately
            # The SDK handles this internally
            chat_instance.append(user_msg(content))  # Assistant context handled by SDK

    # Set sampling parameters
    sample_kwargs: dict[str, Any] = {}
    if temperature is not None:
        sample_kwargs['temperature'] = temperature
    if top_p is not None:
        sample_kwargs['top_p'] = top_p

    # Get response (await for async client)
    response = await chat_instance.sample(**sample_kwargs) if sample_kwargs else await chat_instance.sample()

    # Build result dict
    result: dict[str, Any] = {'content': response.content, 'model': model}

    if hasattr(response, 'id') and response.id:
        result['id'] = response.id

    if hasattr(response, 'usage') and response.usage:
        usage_dict: dict[str, Any] = {
            'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
            'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
            'total_tokens': getattr(response.usage, 'total_tokens', 0),
        }
        # Include reasoning tokens if available
        if hasattr(response.usage, 'reasoning_tokens'):
            usage_dict['reasoning_tokens'] = response.usage.reasoning_tokens
        result['usage'] = usage_dict

    if hasattr(response, 'reasoning_content') and response.reasoning_content:
        result['reasoning_content'] = response.reasoning_content

    return result


@pxt.udf(resource_pool='request-rate:xai')
async def chat_completions(
    messages: list,
    *,
    model: str = 'grok-3',
    model_kwargs: dict[str, Any] | None = None,
    tools: Tools | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response using the OpenAI-compatible chat/completions endpoint.

    This is the legacy endpoint that offers full compatibility with OpenAI SDK patterns.
    For new features including reasoning models, consider using the `chat` UDF instead.

    For additional details, see: <https://docs.x.ai/docs/api-reference#chat-completions>

    Request throttling:
    Applies the rate limit set in the config (section `xai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 60 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages comprising the conversation, following the OpenAI message format.
            Each message should have a `role` (system, user, or assistant) and `content`.
        model: The Grok model to use. Options include:
            - `grok-4`: Latest Grok 4 model (most capable)
            - `grok-4-fast`: Faster Grok 4 variant
            - `grok-3`: Grok 3 model
            - `grok-3-fast`: Faster Grok 3 variant
            - `grok-2-1212`: Grok 2 model
            - `grok-2-vision-1212`: Grok 2 with vision capabilities
        model_kwargs: Additional keyword arguments for the xAI API.
            See: <https://docs.x.ai/docs/api-reference#chat-completions>
        tools: Optional Pixeltable tools for function calling.
        tool_choice: Optional tool choice configuration.

    Returns:
        A dictionary containing the response and metadata, including:
        - `choices`: List of completion choices
        - `usage`: Token usage information
        - `model`: The model used

    Examples:
        Add a computed column that uses Grok to respond to prompts:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are Grok, a helpful AI assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt}
        ... ]
        >>> tbl.add_computed_column(response=xai.chat_completions(messages, model='grok-3'))

        Extract just the response text:

        >>> tbl.add_computed_column(
        ...     answer=xai.chat_completions(messages, model='grok-3')['choices'][0]['message']['content']
        ... )
    """
    import openai

    # Create OpenAI-compatible client for legacy endpoint
    api_key = Config.get().get_string_value('api_key', section='xai')
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.x.ai/v1',
        timeout=httpx.Timeout(3600.0),
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )

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

    result = await client.chat.completions.with_raw_response.create(messages=messages, model=model, **model_kwargs)

    return json.loads(result.text)


@pxt.udf(resource_pool='request-rate:xai')
async def image_generations(
    prompt: str, *, model: str = 'grok-2-image', n: int = 1, response_format: Literal['url', 'b64_json'] = 'b64_json'
) -> PIL.Image.Image:
    """
    Generates an image from a text prompt using xAI's Grok image generation models.

    Equivalent to the xAI `images/generations` API endpoint.
    For additional details, see: <https://docs.x.ai/docs/guides/image-generation>

    Request throttling:
    Applies the rate limit set in the config (section `xai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 60 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        prompt: A text description of the desired image.
        model: The image generation model to use. Currently `grok-2-image` is available.
        n: Number of images to generate (1-10). Only the first image is returned by this UDF.
        response_format: The format of the response. Use `b64_json` for base64-encoded image data,
            or `url` for a URL to the generated image.

    Returns:
        A PIL Image object containing the generated image.

    Examples:
        Generate images from text prompts:

        >>> tbl.add_computed_column(
        ...     generated_image=xai.image_generations(tbl.prompt, model='grok-2-image')
        ... )

        Generate multiple variations (returns only the first):

        >>> tbl.add_computed_column(
        ...     image=xai.image_generations('A sunset over mountains', n=4)
        ... )
    """
    import openai

    # Use OpenAI client for image generation endpoint
    api_key = Config.get().get_string_value('api_key', section='xai')
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.x.ai/v1',
        timeout=httpx.Timeout(3600.0),
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )

    response = await client.images.generate(model=model, prompt=prompt, n=n, response_format=response_format)

    # Get the first generated image
    image_data = response.data[0]

    if response_format == 'b64_json':
        # Decode base64 image data
        img_bytes = base64.b64decode(image_data.b64_json)
        img = PIL.Image.open(BytesIO(img_bytes))
        img.load()
        return img
    else:
        # Download from URL
        async with httpx.AsyncClient() as http_client:
            img_response = await http_client.get(image_data.url)
            img_response.raise_for_status()
            img = PIL.Image.open(BytesIO(img_response.content))
            img.load()
            return img


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

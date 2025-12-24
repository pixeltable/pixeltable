"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) for [xAI](https://x.ai/) Grok models.

Provides integration with xAI's Grok language, vision, and image generation models using the native xAI SDK.

In order to use these UDFs, you must configure your xAI API key either via the `XAI_API_KEY` environment
variable, or as `api_key` in the `xai` section of the Pixeltable config file.
"""

from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal

import httpx
import PIL.Image

import pixeltable as pxt
from pixeltable import env
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

    Uses the native xAI SDK which supports the latest features including the Responses API,
    reasoning models, and vision/image understanding.

    For additional details, see: <https://docs.x.ai/docs/guides/chat>

    Request throttling:
    Applies the rate limit set in the config (section `xai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 60 RPM.

    __Requirements:__

    - `pip install xai-sdk`

    Args:
        messages: A list of messages comprising the conversation. Each message should have
            a `role` (system, user, or assistant) and `content`. Content can be:
            - A string for text-only messages
            - A list of content parts for multimodal messages (text + images), where each part
              has a `type` ('text' or 'image_url') and corresponding data.
        model: The Grok model to use. Options include:
            - `grok-4`: Latest Grok 4 model (most capable, supports vision)
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

        Image understanding (vision):

        >>> messages = [
        ...     {'role': 'user', 'content': [
        ...         {'type': 'text', 'text': 'What is in this image?'},
        ...         {'type': 'image_url', 'image_url': {'url': tbl.image_url, 'detail': 'high'}}
        ...     ]}
        ... ]
        >>> tbl.add_computed_column(response=xai.chat(messages, model='grok-4'))
    """
    from xai_sdk.chat import image, system, user as user_msg

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
            # System messages are text-only
            chat_instance.append(system(content if isinstance(content, str) else str(content)))
        elif role == 'user':
            if isinstance(content, str):
                # Simple text message
                chat_instance.append(user_msg(content))
            elif isinstance(content, list):
                # Multimodal message with text and/or images
                parts: list[Any] = []
                for item in content:
                    item_type = item.get('type', 'text')
                    if item_type == 'text':
                        parts.append(item.get('text', ''))
                    elif item_type == 'image_url':
                        img_data = item.get('image_url', {})
                        img_url = img_data.get('url', '')
                        detail = img_data.get('detail', 'auto')
                        parts.append(image(image_url=img_url, detail=detail))
                chat_instance.append(user_msg(*parts))
            else:
                chat_instance.append(user_msg(str(content)))
        elif role == 'assistant':
            # For assistant messages in conversation history
            if isinstance(content, str):
                chat_instance.append(user_msg(content))
            else:
                chat_instance.append(user_msg(str(content)))

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
async def image_generations(
    prompt: str, *, model: str = 'grok-2-image', image_format: Literal['url', 'base64'] = 'base64'
) -> PIL.Image.Image:
    """
    Generates an image from a text prompt using xAI's Grok image generation models.

    Uses the native xAI SDK `client.image.sample()` method.
    For additional details, see: <https://docs.x.ai/docs/guides/image-generation>

    Request throttling:
    Applies the rate limit set in the config (section `xai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 60 RPM.

    __Requirements:__

    - `pip install xai-sdk`

    Args:
        prompt: A text description of the desired image.
        model: The image generation model to use. Currently `grok-2-image` is available.
        image_format: The format of the response:
            - `base64`: Returns base64-encoded image data (default, faster)
            - `url`: Returns a URL to the generated image on xAI storage

    Returns:
        A PIL Image object containing the generated image.

    Examples:
        Generate images from text prompts:

        >>> tbl.add_computed_column(
        ...     generated_image=xai.image_generations(tbl.prompt, model='grok-2-image')
        ... )
    """
    client = _xai_client()

    # Use native xai_sdk image generation
    response = await client.image.sample(model=model, prompt=prompt, image_format=image_format)

    if image_format == 'base64':
        # response.image contains raw bytes
        img = PIL.Image.open(BytesIO(response.image))
        img.load()
        return img
    else:
        # Download from URL
        async with httpx.AsyncClient() as http_client:
            img_response = await http_client.get(response.url)
            img_response.raise_for_status()
            img = PIL.Image.open(BytesIO(img_response.content))
            img.load()
            return img


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

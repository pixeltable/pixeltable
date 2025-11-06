"""
Pixeltable UDFs that wrap the OpenRouter API.

OpenRouter provides a unified interface to multiple LLM providers. In order to use it,
you must first sign up at https://openrouter.ai, create an API key, and configure it
as described in the Working with OpenRouter tutorial.
"""

from typing import TYPE_CHECKING, Any

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import openai


@register_client('openrouter')
def _(api_key: str, site_url: str | None = None, app_name: str | None = None) -> 'openai.AsyncOpenAI':
    import openai

    # Create default headers for OpenRouter
    default_headers: dict[str, Any] = {}
    if site_url:
        default_headers['HTTP-Referer'] = site_url
    if app_name:
        default_headers['X-Title'] = app_name

    return openai.AsyncOpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key, default_headers=default_headers)


def _openrouter_client() -> 'openai.AsyncOpenAI':
    return Env.get().get_client('openrouter')


@pxt.udf(resource_pool='request-rate:openrouter')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    provider: dict[str, Any] | None = None,
    transforms: list[str] | None = None,
) -> dict:
    """
    Chat Completion API via OpenRouter.

    OpenRouter provides access to multiple LLM providers through a unified API.
    For additional details, see: <https://openrouter.ai/docs>

    Supported models can be found at: <https://openrouter.ai/models>

    Request throttling:
    Applies the rate limit set in the config (section `openrouter`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: ID of the model to use (e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4').
        model_kwargs: Additional OpenAI-compatible parameters.
        tools: List of tools available to the model.
        tool_choice: Controls which (if any) tool is called by the model.
        provider: OpenRouter-specific provider preferences (e.g., {'order': ['Anthropic', 'OpenAI']}).
        transforms: List of message transforms to apply (e.g., ['middle-out']).

    Returns:
        A dictionary containing the response in OpenAI format.

    Examples:
        Basic chat completion:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages,
        ...         model='anthropic/claude-3.5-sonnet'
        ...     )
        ... )

        With provider routing:

        >>> tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages,
        ...         model='anthropic/claude-3.5-sonnet',
        ...         provider={'require_parameters': True, 'order': ['Anthropic']}
        ...     )
        ... )

        With transforms:

        >>> tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages,
        ...         model='openai/gpt-4',
        ...         transforms=['middle-out']  # Optimize for long contexts
        ...     )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('openai')

    # Handle tools if provided
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

    # Prepare OpenRouter-specific parameters for extra_body
    extra_body: dict[str, Any] = {}
    if provider is not None:
        extra_body['provider'] = provider
    if transforms is not None:
        extra_body['transforms'] = transforms

    # Make the API call
    result = await _openrouter_client().chat.completions.create(
        messages=messages, model=model, extra_body=extra_body if extra_body else None, **model_kwargs
    )
    return result.model_dump()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

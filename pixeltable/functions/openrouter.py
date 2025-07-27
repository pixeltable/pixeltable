"""
Pixeltable UDFs that wrap the OpenRouter API.

OpenRouter provides a unified interface to multiple LLM providers. In order to use it,
you must first sign up at https://openrouter.ai, create an API key, and configure it
as described in the Working with OpenRouter tutorial.
"""

from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

# Import OpenAI response converter since OpenRouter uses OpenAI-compatible format
from .openai import _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import openai


@register_client('openrouter')
def _(api_key: str, site_url: Optional[str] = None, app_name: Optional[str] = None) -> 'openai.AsyncOpenAI':
    """
    Register the OpenRouter client with Pixeltable's configuration system.

    Args:
        api_key: Your OpenRouter API key
        site_url: Optional URL for your application (for OpenRouter analytics)
        app_name: Optional name for your application (for OpenRouter analytics)
    """
    import openai

    # Create default headers for OpenRouter
    default_headers = {}
    if site_url:
        default_headers['HTTP-Referer'] = site_url
    if app_name:
        default_headers['X-Title'] = app_name

    return openai.AsyncOpenAI(base_url='https://openrouter.ai/api/v1', api_key=api_key, default_headers=default_headers)


def _openrouter_client() -> 'openai.AsyncOpenAI':
    """Get the registered OpenRouter client."""
    return Env.get().get_client('openrouter')


@pxt.udf(resource_pool='request-rate:openrouter')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[dict[str, Any]] = None,
    provider: Optional[dict[str, Any]] = None,
    transforms: Optional[list[str]] = None,
) -> dict:
    """
    Chat Completion API via OpenRouter.

    OpenRouter provides access to multiple LLM providers through a unified API.
    For additional details, see: https://openrouter.ai/docs

    Supported models can be found at: https://openrouter.ai/models

    __Requirements:__

    - `pip install openai`
    - OpenRouter API key from https://openrouter.ai

    Args:
        messages: A list of messages comprising the conversation so far.
        model: ID of the model to use (e.g., 'anthropic/claude-3.5-sonnet', 'openai/gpt-4').
        model_kwargs: Additional OpenAI-compatible parameters.
        tools: List of tools available to the model (OpenAI format).
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
        model_kwargs['tool_choice'] = tool_choice

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


@pxt.udf(resource_pool='request-rate:openrouter')
async def models() -> list[dict]:
    """
    List available models on OpenRouter.

    Returns a list of model information including pricing, context length,
    and supported features.

    __Requirements:__

    - `pip install openai`
    - OpenRouter API key

    Returns:
        List of dictionaries containing model information.

    Examples:
        Create a table with available models:

        >>> models_table = pxt.create_table('openrouter_models', {'id': pxt.String})
        ... models_table.add_computed_column(models_list=models())
        ... models_table.insert([{'id': '1'}])
        ...
        ... # Access model information
        ... models_df = models_table.select(models_table.models_list).collect()
    """
    Env.get().require_package('openai')

    import httpx

    client = _openrouter_client()

    # OpenRouter's models endpoint
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(
            'https://openrouter.ai/api/v1/models', headers={'Authorization': f'Bearer {client.api_key}'}
        )
        response.raise_for_status()
        data = response.json()

    return data.get('data', [])


# Support tool invocation using OpenAI format
def invoke_tools(tools: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts OpenRouter (OpenAI-compatible) response to Pixeltable tool invocation format."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

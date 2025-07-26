"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the OpenRouter API. In order to use them, you must
first `pip install openai` and configure your OpenRouter credentials.

OpenRouter provides a unified API that gives you access to hundreds of AI models through 
a single endpoint, while automatically handling fallbacks and selecting the most 
cost-effective options.

For more information, see: <https://openrouter.ai/docs>
"""

import json
from typing import TYPE_CHECKING, Any, Optional

import httpx

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names
from .openai import _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import openai


@env.register_client('openrouter')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://openrouter.ai/api/v1',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _openrouter_client() -> 'openai.AsyncOpenAI':
    return env.Env.get().get_client('openrouter')


@pxt.udf
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[dict[str, Any]] = None,
    site_url: Optional[str] = None,
    site_name: Optional[str] = None,
) -> dict:
    """
    Creates a model response for the given chat conversation using OpenRouter.

    Equivalent to the OpenRouter `chat/completions` API endpoint.
    For additional details, see: <https://openrouter.ai/docs/api-reference/chat-completion>

    OpenRouter uses the OpenAI SDK, so you will need to install the `openai` package to use this UDF.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the OpenRouter API documentation.
        model: The model to use for chat completion (e.g., 'openai/gpt-4o', 'anthropic/claude-3-haiku').
        model_kwargs: Additional keyword args for the OpenRouter `chat/completions` API.
            For details on the available parameters, see: <https://openrouter.ai/docs/api-reference/chat-completion>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.
        site_url: Optional site URL for rankings on openrouter.ai.
        site_name: Optional site name for rankings on openrouter.ai.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `openai/gpt-4o-mini` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': tbl.prompt}
            ]
            tbl.add_computed_column(response=chat_completions(messages, model='openai/gpt-4o-mini'))

        Using with site attribution for OpenRouter rankings:

        >>> tbl.add_computed_column(
                response=chat_completions(
                    messages, 
                    model='anthropic/claude-3-haiku',
                    site_url='https://myapp.com',
                    site_name='My App'
                )
            )
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Set up extra headers for OpenRouter attribution
    extra_headers = {}
    if site_url is not None:
        extra_headers['HTTP-Referer'] = site_url
    if site_name is not None:
        extra_headers['X-Title'] = site_name

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
        model_kwargs['parallel_tool_calls'] = False

    result = await _openrouter_client().chat.completions.with_raw_response.create(
        messages=messages, 
        model=model, 
        extra_headers=extra_headers if extra_headers else None,
        **model_kwargs
    )

    return json.loads(result.text)


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenRouter response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__
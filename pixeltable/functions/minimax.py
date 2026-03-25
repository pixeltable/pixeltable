"""
Pixeltable UDFs for MiniMax AI models.

Provides integration with MiniMax's language models for chat completions
via the OpenAI-compatible API. In order to use them, you must first
`pip install openai` and set the `MINIMAX_API_KEY` environment variable
(or configure it in `config.toml`).

For more information about MiniMax models, see: <https://platform.minimaxi.com/document/introduction>
"""

import json
from typing import TYPE_CHECKING, Any

import httpx

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

from .openai import _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import openai


@env.register_client('minimax')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.minimax.io/v1',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _minimax_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('minimax')


@pxt.udf(is_deterministic=False, resource_pool='request-rate:minimax')
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

    Equivalent to the MiniMax `chat/completions` API endpoint (OpenAI-compatible).
    For additional details, see: <https://platform.minimaxi.com/document/chat-completion-v2>

    MiniMax uses the OpenAI SDK, so you will need to install the `openai` package to use this UDF.

    Request throttling:
    Applies the rate limit set in the config (section `minimax`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the MiniMax API documentation.
        model: The model to use for chat completion (e.g., `'MiniMax-M1'`, `'MiniMax-M2.7'`,
            `'MiniMax-M2.5'`, `'MiniMax-M2.5-highspeed'`).
        model_kwargs: Additional keyword args for the MiniMax `chat/completions` API.
            For details on the available parameters, see:
            <https://platform.minimaxi.com/document/chat-completion-v2>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `MiniMax-M2.7` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt},
        ... ]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(messages, model='MiniMax-M2.7')
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

    result = await _minimax_client().chat.completions.with_raw_response.create(
        messages=messages, model=model, **model_kwargs
    )

    return json.loads(result.text)


def invoke_tools(tools: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts a MiniMax response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

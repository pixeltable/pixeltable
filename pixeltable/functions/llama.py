"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Llama API. In order to use them, you must
first `pip install openai` and configure your Llama credentials (typically via
the `LLAMA_API_KEY` environment variable).
"""
import json
import logging
from typing import TYPE_CHECKING, Any, Optional

import httpx

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.utils.code import local_public_names

# _opt is used to convert None to openai.NOT_GIVEN
# _openai_response_to_pxt_tool_calls is used by invoke_tools
from .openai import _opt, _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import openai  # Llama uses an OpenAI-compatible client

_logger = logging.getLogger('pixeltable')


@env.register_client('llama')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.llama.com/compat/v1/',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _llama_client() -> 'openai.AsyncOpenAI':
    return env.Env.get().get_client('llama')


@pxt.udf
async def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,  # Expects pre-formatted tool specs
    tool_choice: Optional[dict] = None,  # Expects pre-formatted dict for specific func choice; use model_kwargs for str like 'auto'
) -> dict:
    """
    Creates a model response for the given chat conversation using the Llama API.

    Equivalent to the Llama `chat_completion` API endpoint, accessed via an OpenAI-compatible interface.
    For additional details, see: <https://llama.developer.meta.com/docs/features/chat-completion/>

    __Requirements:__

    - `pip install openai`
    - An `LLAMA_API_KEY` environment variable.

    Args:
        messages: A list of messages comprising the conversation so far.
        model: ID of the Llama model to use.
        model_kwargs: Additional keyword arguments for the Llama `chat_completions` API.
            These are passed through to the underlying OpenAI-compatible client.
            Use this for parameters like `temperature`, `max_tokens`, `top_p`, etc.,
            and also for `tool_choice` if you need to pass a string like "auto" or "required".
        tools: A list of tool specifications, formatted as expected by the OpenAI API.
               Usually obtained from `ToolsObject.spec`.
        tool_choice: A specific tool choice dictionary (e.g., `{"type": "function", "function": {"name": "my_func"}}`).
                     If you need to pass a string like "auto", use `model_kwargs`.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Basic chat:
        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(response=chat_completions(messages, model='llama3-70b'))

        With model parameters:
        >>> kwargs = {'temperature': 0.7, 'max_tokens': 100}
        ... tbl.add_computed_column(response=chat_completions(messages, model='llama3-70b', model_kwargs=kwargs))

        With tools:
        >>> my_tools = pxt.tools(my_udf1, my_udf2)
        ... tbl.add_computed_column(
        ...    response=chat_completions(
        ...        messages, model='llama3-70b', tools=my_tools.spec, tool_choice=my_tools.choice('my_udf1').spec
        ...    )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    env.Env.get().require_package('openai') # For the openai client library

    # Prioritize tool_choice from model_kwargs if it exists, otherwise use the UDF parameter.
    # This allows passing strings like "auto" via model_kwargs.
    final_tool_choice = model_kwargs.pop('tool_choice', tool_choice)

    result = await _llama_client().chat.completions.with_raw_response.create(
        messages=messages, # type: ignore[arg-type]
        model=model,
        tools=_opt(tools),
        tool_choice=_opt(final_tool_choice),
        **model_kwargs,
    )

    return json.loads(result.text)


def invoke_tools(tools_obj: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts a Llama/OpenAI response dict to Pixeltable tool invocation format and calls `tools_obj._invoke()`."""
    return tools_obj._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

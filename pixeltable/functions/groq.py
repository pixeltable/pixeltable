"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Groq API. In order to use them, you must
first `pip install groq` and configure your Groq credentials, as described in
the [Working with Groq](https://pixeltable.readme.io/docs/working-with-groq) tutorial.
"""

from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable import exprs
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

from .openai import _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import groq


@register_client('groq')
def _(api_key: str) -> 'groq.AsyncGroq':
    import groq

    return groq.AsyncGroq(api_key=api_key)


def _groq_client() -> 'groq.AsyncGroq':
    return Env.get().get_client('groq')


@pxt.udf(resource_pool='request-rate:groq')
async def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str,
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Chat Completion API.

    Equivalent to the Groq `chat/completions` API endpoint.
    For additional details, see: <https://console.groq.com/docs/api-reference#chat-create>

    Request throttling:
    Applies the rate limit set in the config (section `groq`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install groq`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: ID of the model to use. (See overview here: <https://console.groq.com/docs/models>)
        model_kwargs: Additional keyword args for the Groq `chat/completions` API.
            For details on the available parameters, see: <https://console.groq.com/docs/api-reference#chat-create>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `llama3-8b-8192`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(response=chat_completions(messages, model='llama3-8b-8192'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('groq')

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

    result = await _groq_client().chat.completions.create(
        messages=messages,  # type: ignore[arg-type]
        model=model,
        **model_kwargs,
    )
    return result.model_dump()


def invoke_tools(tools: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenAI response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

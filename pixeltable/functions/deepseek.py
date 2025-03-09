
import json
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import httpx

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

from .openai import _opt

if TYPE_CHECKING:
    import openai

@env.register_client('deepseek')
def _(api_key: str) -> 'openai.AsyncOpenAI':
    import openai

    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url='https://api.deepseek.com',
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _deepseek_client() -> 'openai.AsyncOpenAI':
    return env.Env.get().get_client('deepseek')


@pxt.udf
async def chat_completions(
    messages: list,
    *,
    model: str,
    frequency_penalty: Optional[float] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    response_format: Optional[dict] = None,
    stop: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[dict] = None,
    top_p: Optional[float] = None,
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the Deepseek `chat/completions` API endpoint.
    For additional details, see: <https://api-docs.deepseek.com/api/create-chat-completion>

    Deepseek uses the OpenAI SDK, so you will need to install the `openai` package to use this UDF.

    __Requirements:__

    - `pip install openai`

    Args:
        messages: A list of messages to use for chat completion, as described in the Deepseek API documentation.
        model: The model to use for chat completion.

    For details on the other parameters, see: <https://api-docs.deepseek.com/api/create-chat-completion>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `deepseek-chat` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': tbl.prompt}
            ]
            tbl.add_computed_column(response=chat_completions(messages, model='deepseek-chat'))
    """
    if tools is not None:
        tools = [{'type': 'function', 'function': tool} for tool in tools]

    tool_choice_: Union[str, dict, None] = None
    if tool_choice is not None:
        if tool_choice['auto']:
            tool_choice_ = 'auto'
        elif tool_choice['required']:
            tool_choice_ = 'required'
        else:
            assert tool_choice['tool'] is not None
            tool_choice_ = {'type': 'function', 'function': {'name': tool_choice['tool']}}

    extra_body: Optional[dict[str, Any]] = None
    if tool_choice is not None and not tool_choice['parallel_tool_calls']:
        extra_body = {'parallel_tool_calls': False}

    # cast(Any, ...): avoid mypy errors
    result = await _deepseek_client().chat.completions.with_raw_response.create(
        messages=messages,
        model=model,
        frequency_penalty=_opt(frequency_penalty),
        logprobs=_opt(logprobs),
        top_logprobs=_opt(top_logprobs),
        max_tokens=_opt(max_tokens),
        presence_penalty=_opt(presence_penalty),
        response_format=_opt(cast(Any, response_format)),
        stop=_opt(stop),
        temperature=_opt(temperature),
        tools=_opt(cast(Any, tools)),
        tool_choice=_opt(cast(Any, tool_choice_)),
        top_p=_opt(top_p),
        extra_body=extra_body,
    )

    return json.loads(result.text)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

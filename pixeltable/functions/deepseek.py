import json
from typing import TYPE_CHECKING, Any, Optional

import httpx

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

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
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[dict[str, Any]] = None,
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
        model_kwargs: Additional keyword args for the Deepseek `chat/completions` API.
            For details on the available parameters, see: <https://api-docs.deepseek.com/api/create-chat-completion>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

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

    result = await _deepseek_client().chat.completions.with_raw_response.create(
        messages=messages, model=model, **model_kwargs
    )

    return json.loads(result.text)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

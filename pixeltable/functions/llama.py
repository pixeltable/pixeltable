import json
import logging
from typing import TYPE_CHECKING, Any, Optional, cast

import httpx

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names

from .openai import _opt, invoke_tools

if TYPE_CHECKING:
    import openai

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
    messages: list,
    *,
    model: str,
    frequency_penalty: Optional[float] = None,
    # logit_bias: Optional[dict] = None, # Seems unsupported based on OpenAI compatibility docs
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    max_tokens: Optional[int] = None,
    # n: Optional[int] = None, # Seems unsupported
    presence_penalty: Optional[float] = None,
    response_format: Optional[dict] = None,
    # seed: Optional[int] = None, # Seems unsupported
    stop: Optional[list[str]] = None,
    temperature: Optional[float] = None,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[dict] = None,
    top_p: Optional[float] = None,
    # user: Optional[str] = None, # Seems unsupported
) -> dict:
    """
    Creates a model response for the given chat conversation using the Llama API.

    Equivalent to the Llama `chat_completion` API endpoint.
    For additional details, see: <https://llama.developer.meta.com/docs/features/chat-completion/>

    This function uses the Llama API's OpenAI compatibility endpoint.

    __Requirements:__

    - `pip install openai`
    - An `LLAMA_API_KEY` environment variable

    Args:
        messages: A list of messages to use for chat completion, as described in the Llama API documentation.
        model: The model to use for chat completion.

    For details on the other parameters, see: <https://llama.developer.meta.com/docs/features/chat-completion/>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `llama3-70b` to an existing Pixeltable column `tbl.prompt`
        of the table `tbl`:

        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt}
        ... ]
        ... tbl.add_computed_column(response=chat_completions(messages, model='llama3-70b'))
    """
    if tools is not None:
        tools = [{'type': 'function', 'function': tool} for tool in tools]

    tool_choice_: Optional[dict] = tool_choice
    result = await _llama_client().chat.completions.with_raw_response.create(
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
    )

    return json.loads(result.text)



__all__ = local_public_names(__name__) + ['invoke_tools']


def __dir__() -> list[str]:
    return __all__

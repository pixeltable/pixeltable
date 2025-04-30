import json
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import httpx

import pixeltable as pxt
from pixeltable import env
from pixeltable.type_system import JsonType
from pixeltable.utils.code import local_public_names
from pixeltable import exprs
from pixeltable.func import Tools

from .openai import _opt

if TYPE_CHECKING:
    import openai


# TODO: Implement client registration and chat_completions UDF


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

    This function uses the Llama API's OpenAI compatibility endpoint.
    It requires the `openai` package.

    __Requirements:__

    - `pip install openai`
    - An `LLAMA_API_KEY` environment variable or Pixeltable secret.

    Args:
        messages: A list of messages comprising the conversation so far, conforming to the OpenAI message format.
        model: ID of the Llama model to use (e.g., `Llama-3.3-8B-Instruct`).

    For details on supported parameters via the compatibility layer, see the Llama API documentation.
    Supported parameters generally follow the OpenAI `chat/completions` API.

    Returns:
        A dictionary containing the response and other metadata, matching the OpenAI API response structure.

    Examples:
        >>> import pixeltable as pxt
        >>> tbl = pxt.get_table('my_table') # assume tbl has columns 'prompt' and 'image_col'

        **1. Basic Text Completion:**
        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt}
        ... ]
        >>> # Add a computed column using Llama 3.3 8B Instruct
        >>> tbl.add_computed_column(response=pxt.functions.llama.chat_completions(
        ...     messages=messages, model='Llama-3.3-8B-Instruct'))

        **2. Image Understanding:**
        >>> image_prompt = "Describe this image."
        >>> img_messages = [
        ...     {'role': 'user', 'content': [
        ...         {'type': 'text', 'text': image_prompt},
        ...         # Assuming image_col contains PIL Images or paths/URLs recognized by Pixeltable
        ...         {'type': 'image_url', 'image_url': {'url': tbl.image_col.display_url()}}
        ...     ]}
        ... ]
        >>> tbl.add_computed_column(image_desc=pxt.functions.llama.chat_completions(
        ...     messages=img_messages, model='Llama-4-Scout-17B-16E-Instruct-FP8') # Use a vision-capable model
        ...     .choices[0].message.content)

        **3. JSON Output:**
        >>> json_prompt = "Extract the name and date from: Alice went to the park on Friday."
        >>> json_messages = [
        ...     {'role': 'system', 'content': 'Extract information into JSON.'},
        ...     {'role': 'user', 'content': json_prompt}
        ... ]
        >>> tbl.add_computed_column(extracted_json=pxt.functions.llama.chat_completions(
        ...     messages=json_messages,
        ...     model='Llama-3.3-8B-Instruct',
        ...     response_format={'type': 'json_object'} # Request JSON output
        ... ))

        **4. Tool Calling:**
        >>> from pixeltable.functions.util import stock_price # Assume a tool UDF exists
        >>> tool_prompt = "What is the stock price of NVDA?"
        >>> tool_messages = [
        ...     {'role': 'user', 'content': tool_prompt}
        ... ]
        >>> tools = [{
        ...     'name': 'stock_price',
        ...     'description': "Get today's stock price for a ticker.",
        ...     'parameters': {'type': 'object', 'properties': {'ticker': {'type': 'string'}}, 'required': ['ticker']}
        ... }]
        >>> tbl.add_computed_column(tool_response=pxt.functions.llama.chat_completions(
        ...     messages=tool_messages,
        ...     model='Llama-3.3-70B-Instruct', # Use a model capable of tool use
        ...     tools=tools,
        ...     tool_choice='auto'
        ... ))
        >>> # Further steps needed to execute the tool call based on the response...
    """
    if tools is not None:
        tools = [{'type': 'function', 'function': tool} for tool in tools]

    # Llama compatibility layer expects OpenAI format for tool_choice (string or dict)
    # Passing it through directly. The Optional[dict] hint above should map to JsonType.
    # The actual value passed can still be a string or dict from the user.
    tool_choice_: Optional[dict] = tool_choice

    # Note: Llama docs mention specific structure for JSON Schema `response_format`.
    # The OpenAI client might handle the translation, or this might need adjustment
    # if `response_format={'type': 'json_object'}` doesn't work as expected.
    # Starting with standard OpenAI format.

    # Parameters potentially unsupported/ignored by Llama compatibility layer are commented out above.
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


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts a Llama (OpenAI-compatible) response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_llama_response_to_pxt_tool_calls(response))


@pxt.udf
def _llama_response_to_pxt_tool_calls(response: dict) -> Optional[dict]:
    """Helper UDF to convert OpenAI format tool calls to Pixeltable internal format."""
    # Check structure carefully to avoid KeyErrors on unexpected responses
    if not isinstance(response, dict) or 'choices' not in response or not response['choices']:
        return None
    message = response['choices'][0].get('message')
    if not isinstance(message, dict) or 'tool_calls' not in message or message['tool_calls'] is None:
        return None

    openai_tool_calls = message['tool_calls']
    if not isinstance(openai_tool_calls, list):
        return None  # Should be a list

    pxt_tool_calls: dict[str, list[dict[str, Any]]] = {}
    for tool_call in openai_tool_calls:
        if not isinstance(tool_call, dict) or tool_call.get('type') != 'function':
            continue  # Skip non-function calls
        function_call = tool_call.get('function')
        if not isinstance(function_call, dict):
            continue
        tool_name = function_call.get('name')
        arguments_str = function_call.get('arguments')
        if not isinstance(tool_name, str) or not isinstance(arguments_str, str):
            continue  # Skip if name or arguments are missing/wrong type

        try:
            arguments = json.loads(arguments_str)
            if tool_name not in pxt_tool_calls:
                pxt_tool_calls[tool_name] = []
            pxt_tool_calls[tool_name].append({'args': arguments})
        except json.JSONDecodeError:
            _logger.warning(f'Could not decode tool arguments for tool {tool_name}: {arguments_str}')
            continue  # Skip if arguments are not valid JSON

    return pxt_tool_calls if pxt_tool_calls else None


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Anthropic API. In order to use them, you must
first `pip install anthropic` and configure your Anthropic credentials, as described in
the [Working with Anthropic](https://pixeltable.readme.io/docs/working-with-anthropic) tutorial.
"""

from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union

import tenacity

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import anthropic


@env.register_client('anthropic')
def _(api_key: str) -> 'anthropic.Anthropic':
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def _anthropic_client() -> 'anthropic.Anthropic':
    return env.Env.get().get_client('anthropic')


def _retry(fn: Callable) -> Callable:
    import anthropic
    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(anthropic.RateLimitError),
        wait=tenacity.wait_random_exponential(multiplier=1, max=60),
        stop=tenacity.stop_after_attempt(20),
    )(fn)


@pxt.udf
def messages(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int = 1024,
    metadata: Optional[dict[str, Any]] = None,
    stop_sequences: Optional[list[str]] = None,
    system: Optional[str] = None,
    temperature: Optional[float] = None,
    tool_choice: Optional[pxt.Json] = None,
    tools: Optional[list[dict]] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> dict:
    """
    Create a Message.

    Equivalent to the Anthropic `messages` API endpoint.
    For additional details, see: <https://docs.anthropic.com/en/api/messages>

    __Requirements:__

    - `pip install anthropic`

    Args:
        messages: Input messages.
        model: The model that will complete your prompt.

    For details on the other parameters, see: <https://docs.anthropic.com/en/api/messages>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `claude-3-haiku-20240307`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> msgs = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl['response'] = messages(msgs, model='claude-3-haiku-20240307')
    """
    if tools is not None:
        # Reformat `tools` into Anthropic format
        tools = [
            {
                'name': tool['name'],
                'description': tool['description'],
                'input_schema': {
                    'type': 'object',
                    'properties': tool['parameters']['properties'],
                    'required': tool['required'],
                },
            }
            for tool in tools
        ]

    return _retry(_anthropic_client().messages.create)(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        metadata=_opt(metadata),
        stop_sequences=_opt(stop_sequences),
        system=_opt(system),
        temperature=_opt(temperature),
        tool_choice=_opt(tool_choice),
        tools=_opt(tools),
        top_k=_opt(top_k),
        top_p=_opt(top_p),
    ).dict()


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an Anthropic response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_anthropic_response_to_pxt_tool_calls(response))


@pxt.udf
def _anthropic_response_to_pxt_tool_calls(response: dict) -> Optional[dict]:
    anthropic_tool_calls = [r for r in response['content'] if r['type'] == 'tool_use']
    if len(anthropic_tool_calls) > 0:
        return {
            tool_call['name']: {
                'args': tool_call['input']
            }
            for tool_call in anthropic_tool_calls
        }
    return None


_T = TypeVar('_T')


def _opt(arg: _T) -> Union[_T, 'anthropic.NotGiven']:
    import anthropic
    return arg if arg is not None else anthropic.NOT_GIVEN


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

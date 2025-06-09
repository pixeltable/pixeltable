"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Anthropic API. In order to use them, you must
first `pip install anthropic` and configure your Anthropic credentials, as described in
the [Working with Anthropic](https://pixeltable.readme.io/docs/working-with-anthropic) tutorial.
"""

import datetime
import json
import logging
from typing import TYPE_CHECKING, Any, Iterable, Optional, cast

import httpx

import pixeltable as pxt
from pixeltable import env, exprs
from pixeltable.func import Tools
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import anthropic

_logger = logging.getLogger('pixeltable')


@env.register_client('anthropic')
def _(api_key: str) -> 'anthropic.AsyncAnthropic':
    import anthropic

    return anthropic.AsyncAnthropic(
        api_key=api_key,
        # recommended to increase limits for async client to avoid connection errors
        http_client=httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=100, max_connections=500)),
    )


def _anthropic_client() -> 'anthropic.AsyncAnthropic':
    return env.Env.get().get_client('anthropic')


class AnthropicRateLimitsInfo(env.RateLimitsInfo):
    def __init__(self) -> None:
        super().__init__(self._get_request_resources)

    def _get_request_resources(self, messages: dict, max_tokens: int) -> dict[str, int]:
        input_len = 0
        for message in messages:
            if 'role' in message:
                input_len += len(message['role'])
            if 'content' in message:
                input_len += len(message['content'])
        return {'requests': 1, 'input_tokens': int(input_len / 4), 'output_tokens': max_tokens}

    def get_retry_delay(self, exc: Exception) -> Optional[float]:
        import anthropic

        # deal with timeouts separately, they don't come with headers
        if isinstance(exc, anthropic.APITimeoutError):
            return 1.0

        if not isinstance(exc, anthropic.APIStatusError):
            return None
        _logger.debug(f'headers={exc.response.headers}')
        should_retry_str = exc.response.headers.get('x-should-retry', '')
        if should_retry_str.lower() != 'true':
            return None
        retry_after_str = exc.response.headers.get('retry-after', '1')
        return int(retry_after_str)


@pxt.udf
async def messages(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: int,
    model_kwargs: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
    tool_choice: Optional[dict[str, Any]] = None,
) -> dict:
    """
    Create a Message.

    Equivalent to the Anthropic `messages` API endpoint.
    For additional details, see: <https://docs.anthropic.com/en/api/messages>

    Request throttling:
    Uses the rate limit-related headers returned by the API to throttle requests adaptively, based on available
    request and token capacity. No configuration is necessary.

    __Requirements:__

    - `pip install anthropic`

    Args:
        messages: Input messages.
        model: The model that will complete your prompt.
        model_kwargs: Additional keyword args for the Anthropic `messages` API.
            For details on the available parameters, see: <https://docs.anthropic.com/en/api/messages>
        tools: An optional list of Pixeltable tools to use for the request.
        tool_choice: An optional tool choice configuration.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `claude-3-5-sonnet-20241022`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> msgs = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(response=messages(msgs, model='claude-3-5-sonnet-20241022'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    if tools is not None:
        # Reformat `tools` into Anthropic format
        model_kwargs['tools'] = [
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

    if tool_choice is not None:
        if tool_choice['auto']:
            model_kwargs['tool_choice'] = {'type': 'auto'}
        elif tool_choice['required']:
            model_kwargs['tool_choice'] = {'type': 'any'}
        else:
            assert tool_choice['tool'] is not None
            model_kwargs['tool_choice'] = {'type': 'tool', 'name': tool_choice['tool']}
        if not tool_choice['parallel_tool_calls']:
            model_kwargs['tool_choice']['disable_parallel_tool_use'] = True

    # make sure the pool info exists prior to making the request
    resource_pool_id = f'rate-limits:anthropic:{model}'
    rate_limits_info = env.Env.get().get_resource_pool_info(resource_pool_id, AnthropicRateLimitsInfo)
    assert isinstance(rate_limits_info, env.RateLimitsInfo)

    # TODO: timeouts should be set system-wide and be user-configurable
    from anthropic.types import MessageParam

    result = await _anthropic_client().messages.with_raw_response.create(
        messages=cast(Iterable[MessageParam], messages), model=model, max_tokens=max_tokens, **model_kwargs
    )

    requests_limit_str = result.headers.get('anthropic-ratelimit-requests-limit')
    requests_limit = int(requests_limit_str) if requests_limit_str is not None else None
    requests_remaining_str = result.headers.get('anthropic-ratelimit-requests-remaining')
    requests_remaining = int(requests_remaining_str) if requests_remaining_str is not None else None
    requests_reset_str = result.headers.get('anthropic-ratelimit-requests-reset')
    requests_reset = datetime.datetime.fromisoformat(requests_reset_str.replace('Z', '+00:00'))
    input_tokens_limit_str = result.headers.get('anthropic-ratelimit-input-tokens-limit')
    input_tokens_limit = int(input_tokens_limit_str) if input_tokens_limit_str is not None else None
    input_tokens_remaining_str = result.headers.get('anthropic-ratelimit-input-tokens-remaining')
    input_tokens_remaining = int(input_tokens_remaining_str) if input_tokens_remaining_str is not None else None
    input_tokens_reset_str = result.headers.get('anthropic-ratelimit-input-tokens-reset')
    input_tokens_reset = datetime.datetime.fromisoformat(input_tokens_reset_str.replace('Z', '+00:00'))
    output_tokens_limit_str = result.headers.get('anthropic-ratelimit-output-tokens-limit')
    output_tokens_limit = int(output_tokens_limit_str) if output_tokens_limit_str is not None else None
    output_tokens_remaining_str = result.headers.get('anthropic-ratelimit-output-tokens-remaining')
    output_tokens_remaining = int(output_tokens_remaining_str) if output_tokens_remaining_str is not None else None
    output_tokens_reset_str = result.headers.get('anthropic-ratelimit-output-tokens-reset')
    output_tokens_reset = datetime.datetime.fromisoformat(output_tokens_reset_str.replace('Z', '+00:00'))
    retry_after_str = result.headers.get('retry-after')
    if retry_after_str is not None:
        _logger.debug(f'retry-after: {retry_after_str}')

    rate_limits_info.record(
        requests=(requests_limit, requests_remaining, requests_reset),
        input_tokens=(input_tokens_limit, input_tokens_remaining, input_tokens_reset),
        output_tokens=(output_tokens_limit, output_tokens_remaining, output_tokens_reset),
    )

    result_dict = json.loads(result.text)
    return result_dict


@messages.resource_pool
def _(model: str) -> str:
    return f'rate-limits:anthropic:{model}'


def invoke_tools(tools: Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an Anthropic response dict to Pixeltable tool invocation format and calls `tools._invoke()`."""
    return tools._invoke(_anthropic_response_to_pxt_tool_calls(response))


@pxt.udf
def _anthropic_response_to_pxt_tool_calls(response: dict) -> Optional[dict]:
    anthropic_tool_calls = [r for r in response['content'] if r['type'] == 'tool_use']
    if len(anthropic_tool_calls) == 0:
        return None
    pxt_tool_calls: dict[str, list[dict[str, Any]]] = {}
    for tool_call in anthropic_tool_calls:
        tool_name = tool_call['name']
        if tool_name not in pxt_tool_calls:
            pxt_tool_calls[tool_name] = []
        pxt_tool_calls[tool_name].append({'args': tool_call['input']})
    return pxt_tool_calls


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

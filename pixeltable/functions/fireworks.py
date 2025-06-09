"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Fireworks AI API. In order to use them, you must
first `pip install fireworks-ai` and configure your Fireworks AI credentials, as described in
the [Working with Fireworks](https://pixeltable.readme.io/docs/working-with-fireworks) tutorial.
"""

from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable import env
from pixeltable.config import Config
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import fireworks.client  # type: ignore[import-untyped]


@env.register_client('fireworks')
def _(api_key: str) -> 'fireworks.client.Fireworks':
    import fireworks.client

    return fireworks.client.Fireworks(api_key=api_key)


def _fireworks_client() -> 'fireworks.client.Fireworks':
    return env.Env.get().get_client('fireworks')


@pxt.udf(resource_pool='request-rate:fireworks')
async def chat_completions(
    messages: list[dict[str, str]], *, model: str, model_kwargs: Optional[dict[str, Any]] = None
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the Fireworks AI `chat/completions` API endpoint.
    For additional details, see: <https://docs.fireworks.ai/api-reference/post-chatcompletions>

    Request throttling:
    Applies the rate limit set in the config (section `fireworks`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install fireworks-ai`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The name of the model to use.
        model_kwargs: Additional keyword args for the Fireworks `chat_completions` API. For details on the available
            parameters, see: <https://docs.fireworks.ai/api-reference/post-chatcompletions>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `accounts/fireworks/models/mixtral-8x22b-instruct`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(
        ...     response=chat_completions(messages, model='accounts/fireworks/models/mixtral-8x22b-instruct')
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    # for debugging purposes:
    # res_sync = _fireworks_client().chat.completions.create(model=model, messages=messages, **kwargs_not_none)
    # res_sync_dict = res_sync.dict()

    if 'request_timeout' not in model_kwargs:
        model_kwargs['request_timeout'] = Config.get().get_int_value('timeout', section='fireworks') or 600
    # TODO: this timeout doesn't really work, I think it only applies to returning the stream, but not to the timing
    # of the chunks; addressing this would require a timeout for the task running this udf
    stream = _fireworks_client().chat.completions.acreate(model=model, messages=messages, **model_kwargs)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)

    res = {
        'id': chunks[0].id,
        'object': 'chat.completion',
        'created': chunks[0].created,
        'model': chunks[0].model,
        'choices': [
            {
                'index': 0,
                'message': {
                    'role': None,
                    'content': '',
                    'tool_calls': None,
                    'tool_call_id': None,
                    'function': None,
                    'name': None,
                },
                'finish_reason': None,
                'logprobs': None,
                'raw_output': None,
            }
        ],
        'usage': {},
    }
    for chunk in chunks:
        d = chunk.dict()
        if 'usage' in d and d['usage'] is not None:
            res['usage'] = d['usage']
        if chunk.choices[0].finish_reason is not None:
            res['choices'][0]['finish_reason'] = chunk.choices[0].finish_reason
        if chunk.choices[0].delta.role is not None:
            res['choices'][0]['message']['role'] = chunk.choices[0].delta.role
        if chunk.choices[0].delta.content is not None:
            res['choices'][0]['message']['content'] += chunk.choices[0].delta.content
        if chunk.choices[0].delta.tool_calls is not None:
            res['choices'][0]['message']['tool_calls'] = chunk.choices[0].delta.tool_calls
        if chunk.choices[0].delta.function is not None:
            res['choices'][0]['message']['function'] = chunk.choices[0].delta.function
    return res


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

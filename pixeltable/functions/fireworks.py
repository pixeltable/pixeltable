"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Fireworks AI API. In order to use them, you must
first `pip install fireworks-ai` and configure your Fireworks AI credentials, as described in
the [Working with Fireworks](https://pixeltable.readme.io/docs/working-with-fireworks) tutorial.
"""

from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from pixeltable import env
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import fireworks.client  # type: ignore[import-untyped]


@env.register_client('fireworks')
def _(api_key: str) -> 'fireworks.client.Fireworks':
    import fireworks.client

    return fireworks.client.Fireworks(api_key=api_key)


def _fireworks_client() -> 'fireworks.client.Fireworks':
    return env.Env.get().get_client('fireworks')


@pxt.udf
def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str,
    max_tokens: Optional[int] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    temperature: Optional[float] = None,
) -> dict:
    """
    Creates a model response for the given chat conversation.

    Equivalent to the Fireworks AI `chat/completions` API endpoint.
    For additional details, see: [https://docs.fireworks.ai/api-reference/post-chatcompletions](https://docs.fireworks.ai/api-reference/post-chatcompletions)

    __Requirements:__

    - `pip install fireworks-ai`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The name of the model to use.

    For details on the other parameters, see: [https://docs.fireworks.ai/api-reference/post-chatcompletions](https://docs.fireworks.ai/api-reference/post-chatcompletions)

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `accounts/fireworks/models/mixtral-8x22b-instruct`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl['response'] = chat_completions(messages, model='accounts/fireworks/models/mixtral-8x22b-instruct')
    """
    kwargs = {'max_tokens': max_tokens, 'top_k': top_k, 'top_p': top_p, 'temperature': temperature}
    kwargs_not_none = {k: v for k, v in kwargs.items() if v is not None}
    return _fireworks_client().chat.completions.create(model=model, messages=messages, **kwargs_not_none).dict()


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

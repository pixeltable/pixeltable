"""
Pixeltable UDFs
that wrap various endpoints from the Fireworks AI API. In order to use them, you must
first `pip install fireworks-ai` and configure your Fireworks AI credentials, as described in
the [Working with Fireworks](https://docs.pixeltable.com/howto/providers/working-with-fireworks) tutorial.
"""

from typing import TYPE_CHECKING, Any, Iterable, cast

import pixeltable as pxt
from pixeltable import env
from pixeltable.config import Config
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import fireworks
    from fireworks.types.shared_params.chat_message import ChatMessage


@env.register_client('fireworks', credential_param='api_key')
def _(api_key: str) -> 'fireworks.AsyncFireworks':
    import fireworks

    return fireworks.AsyncFireworks(api_key=api_key)


def _fireworks_client() -> 'fireworks.AsyncFireworks':
    return get_runtime().get_client('fireworks')


@pxt.udf(is_deterministic=False, resource_pool='request-rate:fireworks')
async def chat_completions(
    messages: list[dict[str, str]], *, model: str, model_kwargs: dict[str, Any] | None = None
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
        Add a computed column that applies the model `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages,
        ...         model='accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b',
        ...     )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    if 'timeout' not in model_kwargs:
        model_kwargs['timeout'] = Config.get().get_int_value('timeout', section='fireworks') or 600

    result = await _fireworks_client().chat.completions.create(
        model=model, messages=cast(Iterable['ChatMessage'], messages), **model_kwargs
    )
    return result.model_dump(mode='json')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

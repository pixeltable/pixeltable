"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Groq API. In order to use them, you must
first `pip install groq` and configure your Groq credentials, as described in
the [Working with Groq](https://pixeltable.readme.io/docs/working-with-groq) tutorial.
"""

from typing import TYPE_CHECKING, Any, Optional

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.utils.code import local_public_names

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
    messages: list[dict[str, str]], *, model: str, model_kwargs: Optional[dict[str, Any]] = None
) -> dict:
    """
    Chat Completion API.

    Equivalent to the Mistral AI `chat/completions` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/chat>

    Request throttling:
    Applies the rate limit set in the config (section `mistral`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install mistralai`

    Args:
        messages: The prompt(s) to generate completions for.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)
        model_kwargs: Additional keyword args for the Mistral `chat/completions` API.
            For details on the available parameters, see: <https://docs.mistral.ai/api/#tag/chat>

    For details on the other parameters, see: <https://docs.mistral.ai/api/#tag/chat>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistral-latest-small`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl.add_computed_column(response=completions(messages, model='mistral-latest-small'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('groq')
    result = await _groq_client().chat.completions.create(
        messages=messages,  # type: ignore[arg-type]
        model=model,
        **model_kwargs,
    )
    return result.dict()


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

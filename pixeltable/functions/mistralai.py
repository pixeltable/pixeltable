"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Mistral AI API. In order to use them, you must
first `pip install mistralai` and configure your Mistral AI credentials, as described in
the [Working with Mistral AI](https://pixeltable.readme.io/docs/working-with-mistralai) tutorial.
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.env import Env, register_client
from pixeltable.func.signature import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import mistralai


@register_client('mistral')
def _(api_key: str) -> 'mistralai.Mistral':
    import mistralai

    return mistralai.Mistral(api_key=api_key)


def _mistralai_client() -> 'mistralai.Mistral':
    return Env.get().get_client('mistral')


@pxt.udf(resource_pool='request-rate:mistral')
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

    Env.get().require_package('mistralai')
    result = await _mistralai_client().chat.complete_async(
        messages=messages,  # type: ignore[arg-type]
        model=model,
        **model_kwargs,
    )
    return result.dict()


@pxt.udf(resource_pool='request-rate:mistral')
async def fim_completions(prompt: str, *, model: str, model_kwargs: Optional[dict[str, Any]] = None) -> dict:
    """
    Fill-in-the-middle Completion API.

    Equivalent to the Mistral AI `fim/completions` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/fim>

    Request throttling:
    Applies the rate limit set in the config (section `mistral`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install mistralai`

    Args:
        prompt: The text/code to complete.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)
        model_kwargs: Additional keyword args for the Mistral `fim/completions` API.
            For details on the available parameters, see: <https://docs.mistral.ai/api/#tag/fim>

    For details on the other parameters, see: <https://docs.mistral.ai/api/#tag/fim>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `codestral-latest`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl.add_computed_column(response=completions(tbl.prompt, model='codestral-latest'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('mistralai')
    result = await _mistralai_client().fim.complete_async(prompt=prompt, model=model, **model_kwargs)
    return result.dict()


_embedding_dimensions_cache: dict[str, int] = {'mistral-embed': 1024}


@pxt.udf(batch_size=16, resource_pool='request-rate:mistral')
async def embeddings(input: Batch[str], *, model: str) -> Batch[pxt.Array[(None,), pxt.Float]]:  # noqa: RUF029
    """
    Embeddings API.

    Equivalent to the Mistral AI `embeddings` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/embeddings>

    Request throttling:
    Applies the rate limit set in the config (section `mistral`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install mistralai`

    Args:
        input: Text to embed.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)

    Returns:
        An array representing the application of the given embedding to `input`.
    """
    Env.get().require_package('mistralai')
    result = _mistralai_client().embeddings.create(inputs=input, model=model)
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str) -> ts.ArrayType:
    dimensions = _embedding_dimensions_cache.get(model)  # `None` if unknown model
    return ts.ArrayType((dimensions,), dtype=ts.FloatType())


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

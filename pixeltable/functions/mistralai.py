"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs)
that wrap various endpoints from the Mistral AI API. In order to use them, you must
first `pip install mistralai` and configure your Mistral AI credentials, as described in
the [Working with Mistral AI](https://pixeltable.readme.io/docs/working-with-mistralai) tutorial.
"""

from typing import TYPE_CHECKING, Optional, TypeVar, Union

import numpy as np

import pixeltable as pxt
from pixeltable.env import Env, register_client
from pixeltable.func.signature import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    import mistralai.types.basemodel


@register_client('mistral')
def _(api_key: str) -> 'mistralai.Mistral':
    import mistralai
    return mistralai.Mistral(api_key=api_key)


def _mistralai_client() -> 'mistralai.Mistral':
    return Env.get().get_client('mistral')


@pxt.udf
def chat_completions(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
    response_format: Optional[dict] = None,
    safe_prompt: Optional[bool] = False,
) -> dict:
    """
    Chat Completion API.

    Equivalent to the Mistral AI `chat/completions` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/chat>

    __Requirements:__

    - `pip install mistralai`

    Args:
        messages: The prompt(s) to generate completions for.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)

    For details on the other parameters, see: <https://docs.mistral.ai/api/#tag/chat>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `mistral-latest-small`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        ... tbl['response'] = completions(messages, model='mistral-latest-small')
    """
    Env.get().require_package('mistralai')
    return _mistralai_client().chat.complete(
        messages=messages,  # type: ignore[arg-type]
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=_opt(max_tokens),
        min_tokens=_opt(min_tokens),
        stop=stop,
        random_seed=_opt(random_seed),
        response_format=response_format,  # type: ignore[arg-type]
        safe_prompt=safe_prompt,
    ).dict()


@pxt.udf
def fim_completions(
    prompt: str,
    *,
    model: str,
    temperature: Optional[float] = 0.7,
    top_p: Optional[float] = 1.0,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    random_seed: Optional[int] = None,
    suffix: Optional[str] = None,
) -> dict:
    """
    Fill-in-the-middle Completion API.

    Equivalent to the Mistral AI `fim/completions` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/fim>

    __Requirements:__

    - `pip install mistralai`

    Args:
        prompt: The text/code to complete.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)

    For details on the other parameters, see: <https://docs.mistral.ai/api/#tag/fim>

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `codestral-latest`
        to an existing Pixeltable column `tbl.prompt` of the table `tbl`:

        >>> tbl['response'] = completions(tbl.prompt, model='codestral-latest')
    """
    Env.get().require_package('mistralai')
    return _mistralai_client().fim.complete(
        prompt=prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=_opt(max_tokens),
        min_tokens=_opt(min_tokens),
        stop=stop,
        random_seed=_opt(random_seed),
        suffix=_opt(suffix)
    ).dict()


_embedding_dimensions_cache: dict[str, int] = {
    'mistral-embed': 1024
}


@pxt.udf(batch_size=16)
def embeddings(input: Batch[str], *, model: str) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Embeddings API.

    Equivalent to the Mistral AI `embeddings` API endpoint.
    For additional details, see: <https://docs.mistral.ai/api/#tag/embeddings>

    __Requirements:__

    - `pip install mistralai`

    Args:
        input: Text to embed.
        model: ID of the model to use. (See overview here: <https://docs.mistral.ai/getting-started/models/>)

    Returns:
        An array representing the application of the given embedding to `input`.
    """
    Env.get().require_package('mistralai')
    result = _mistralai_client().embeddings.create(
        inputs=input,
        model=model,
    )
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str) -> pxt.ArrayType:
    dimensions = _embedding_dimensions_cache.get(model)  # `None` if unknown model
    return pxt.ArrayType((dimensions,), dtype=pxt.FloatType())


_T = TypeVar('_T')


def _opt(arg: Optional[_T]) -> Union[_T, 'mistralai.types.basemodel.Unset']:
    from mistralai.types import UNSET
    return arg if arg is not None else UNSET


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

"""
Pixeltable UDFs that wrap Databricks Model Serving and Mosaic AI agents.

In order to use them, you must first ``pip install openai`` and configure your Databricks
credentials, as described in the
[Working with Databricks](https://docs.pixeltable.com/howto/providers/working-with-databricks) tutorial.
"""

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.env import Env, register_client
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64

from .openai import _openai_response_to_pxt_tool_calls

if TYPE_CHECKING:
    import openai


@register_client('databricks', credential_param='token')
def _(host: str, token: str) -> 'openai.AsyncOpenAI':
    """Create an OpenAI-compatible client pointed at Databricks Model Serving."""
    Env.get().require_package('openai')
    import openai

    base_url = f'{host.rstrip("/")}/serving-endpoints'
    return openai.AsyncOpenAI(api_key=token, base_url=base_url)


def _databricks_client() -> 'openai.AsyncOpenAI':
    return get_runtime().get_client('databricks')


_embedding_dimensions: dict[str, int] = {'databricks-gte-large-en': 1024, 'databricks-bge-large-en': 1024}


@pxt.udf(is_deterministic=False, resource_pool='request-rate:databricks')
async def chat_completions(
    messages: list[dict[str, Any]],
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
) -> dict:
    """
    Chat completions via Databricks Model Serving or Foundation Models.

    Equivalent to the Databricks OpenAI-compatible ``chat/completions`` API.
    For additional details, see:
    <https://docs.databricks.com/en/machine-learning/foundation-models/index.html>

    Request throttling:
    Applies the rate limit set in the config (section ``databricks``, key ``rate_limit``). If no rate
    limit is configured, uses a default RPM for the ``request-rate:databricks`` pool.

    Pass a foundation model name (e.g. ``databricks-meta-llama-3-3-70b-instruct``) or any
    deployed Model Serving endpoint name — including Mosaic AI agents (Knowledge Assistant,
    Multi-Agent Supervisor, custom MLflow agents).

    __Requirements:__

    - ``pip install openai``

    Args:
        messages: OpenAI-compatible chat messages.
        model: Foundation model or serving endpoint name.
        model_kwargs: Additional kwargs for the chat completions API.
        tools: Optional function tools.
        tool_choice: Optional tool choice configuration.

    Returns:
        OpenAI-compatible chat completion response as a dict.

    Examples:
        >>> tbl.add_computed_column(
        ...     summary=chat_completions(
        ...         [{'role': 'user', 'content': tbl.text}],
        ...         model='databricks-meta-llama-3-3-70b-instruct',
        ...     )['choices'][0]['message']['content']
        ... )

        Call a deployed Mosaic AI agent endpoint by name:

        >>> tbl.add_computed_column(
        ...     answer=chat_completions(
        ...         [{'role': 'user', 'content': tbl.question}],
        ...         model='my-rag-agent-endpoint',
        ...     )['choices'][0]['message']['content']
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('openai')

    if tools is not None:
        model_kwargs['tools'] = [{'type': 'function', 'function': tool} for tool in tools]

    if tool_choice is not None:
        if tool_choice['auto']:
            model_kwargs['tool_choice'] = 'auto'
        elif tool_choice['required']:
            model_kwargs['tool_choice'] = 'required'
        else:
            assert tool_choice['tool'] is not None
            model_kwargs['tool_choice'] = {'type': 'function', 'function': {'name': tool_choice['tool']}}

    if tool_choice is not None and not tool_choice['parallel_tool_calls']:
        model_kwargs['parallel_tool_calls'] = False

    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get('content')
        if isinstance(content, list):
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get('type') == 'image_url'
                    and isinstance(part.get('image_url'), PIL.Image.Image)
                ):
                    b64_encoded_image = to_base64(part['image_url'], format='png')
                    part['image_url'] = {'url': f'data:image/png;base64,{b64_encoded_image}'}

    result = await _databricks_client().chat.completions.create(
        messages=messages,  # type: ignore[arg-type]
        model=model,
        **model_kwargs,
    )
    return result.model_dump()


@pxt.udf(batch_size=32, is_deterministic=False, resource_pool='request-rate:databricks')
async def embeddings(
    input: Batch[str], *, model: str, model_kwargs: dict[str, Any] | None = None
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Create embedding vectors via Databricks embedding endpoints.

    Equivalent to the Databricks OpenAI-compatible ``embeddings`` API.
    For additional details, see:
    <https://docs.databricks.com/en/machine-learning/foundation-models/index.html>

    Request throttling:
    Applies the rate limit set in the config (section ``databricks``, key ``rate_limit``). If no rate
    limit is configured, uses a default RPM for the ``request-rate:databricks`` pool.

    __Requirements:__

    - ``pip install openai``

    Args:
        input: Text strings to embed.
        model: Embedding model or endpoint name (e.g. ``databricks-gte-large-en``).
        model_kwargs: Additional kwargs for the embeddings API.

    Returns:
        Embedding vectors as float arrays.

    Examples:
        >>> tbl.add_computed_column(
        ...     embed=embeddings(tbl.text, model='databricks-gte-large-en')
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('openai')
    result = await _databricks_client().embeddings.create(
        input=input, model=model, encoding_format='float', **model_kwargs
    )
    return [np.array(data.embedding, dtype=np.float64) for data in result.data]


@embeddings.conditional_return_type
def _(model: str, model_kwargs: dict[str, Any] | None = None) -> ts.ArrayType:
    dimensions: int | None = None
    if model_kwargs is not None:
        dimensions = model_kwargs.get('dimensions')
    if dimensions is None:
        if model not in _embedding_dimensions:
            return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)
        dimensions = _embedding_dimensions[model]
    return ts.ArrayType((dimensions,), dtype=ts.FloatType(), nullable=False)


@pxt.udf(is_deterministic=False, resource_pool='request-rate:databricks')
async def responses(
    input: list[dict[str, Any]],
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Responses API for Databricks Supervisor API, Apps agents, and ResponsesAgent endpoints.

    Equivalent to the Databricks OpenAI-compatible ``responses`` API.
    For additional details, see:
    <https://docs.databricks.com/en/generative-ai/agent-framework/responses-agent.html>

    Request throttling:
    Applies the rate limit set in the config (section ``databricks``, key ``rate_limit``). If no rate
    limit is configured, uses a default RPM for the ``request-rate:databricks`` pool.

    Use ``model='apps/<app-name>'`` for Databricks Apps agents, or pass hosted tools for
    the Supervisor API (Genie spaces, Vector Search indexes, UC tables, etc.).

    __Requirements:__

    - ``pip install openai``

    Args:
        input: Responses API input items.
        model: Model or endpoint name.
        model_kwargs: Additional kwargs (e.g. ``extra_body`` for trace_destination).
        tools: Hosted tools for Supervisor API agent loops.

    Returns:
        Responses API result as a dict.

    Examples:
        >>> tbl.add_computed_column(
        ...     analysis=responses(
        ...         input=[{'role': 'user', 'content': tbl.text}],
        ...         model='databricks-claude-sonnet-4-5',
        ...         tools=[
        ...             {'type': 'table', 'table': {'name': 'catalog.schema.reviews'}}
        ...         ],
        ...     )['output_text']
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}
    if tools is not None:
        model_kwargs['tools'] = tools

    Env.get().require_package('openai')
    result = await _databricks_client().responses.create(
        input=input,  # type: ignore[arg-type]
        model=model,
        **model_kwargs,
    )
    return result.model_dump()


def invoke_tools(tools: pxt.func.Tools, response: exprs.Expr) -> exprs.InlineDict:
    """Converts an OpenAI response dict to Pixeltable tool invocation format and calls ``tools._invoke()``."""
    return tools._invoke(_openai_response_to_pxt_tool_calls(response))


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

"""
Pixeltable UDFs that wrap the LiteLLM SDK.

LiteLLM provides a unified interface to 100+ LLM providers (Anthropic, Bedrock, Vertex AI,
Cohere, Mistral, etc.) through a single `completion()` call. The provider is specified via
the model string (e.g. `anthropic/claude-sonnet-4-5`, `bedrock/anthropic.claude-v2`).

In order to use these UDFs, you must first `pip install litellm` and set the appropriate
provider-specific environment variables (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`).
See https://docs.litellm.ai/docs/providers for the full list of supported providers.
"""

from typing import Any

import pixeltable as pxt
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names


@pxt.udf(is_deterministic=False, resource_pool='request-rate:litellm')
async def chat_completions(
    messages: list,
    *,
    model: str,
    model_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given chat conversation using LiteLLM.

    LiteLLM routes the request to the correct provider based on the model string.
    For example, `anthropic/claude-sonnet-4-5` routes to Anthropic, `bedrock/anthropic.claude-v2`
    routes to AWS Bedrock, and `vertex_ai/gemini-pro` routes to Google Vertex AI.

    For a full list of supported models and providers, see: <https://docs.litellm.ai/docs/providers>

    Request throttling:
    Applies the rate limit set in the config (section `litellm`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install litellm`

    Args:
        messages: A list of messages comprising the conversation so far.
        model: The model to use, prefixed with the provider name
            (e.g. `'anthropic/claude-sonnet-4-5'`, `'openai/gpt-4o'`, `'bedrock/anthropic.claude-v2'`).
        model_kwargs: Additional keyword args passed to `litellm.acompletion()`.
            For details, see: <https://docs.litellm.ai/docs/completion/input>

    Returns:
        A dictionary containing the response and other metadata in OpenAI-compatible format.

    Examples:
        Add a computed column that applies Claude via LiteLLM to an existing Pixeltable column `tbl.prompt`:

        >>> messages = [{'role': 'user', 'content': tbl.prompt}]
        >>> tbl.add_computed_column(
        ...     response=chat_completions(messages, model='anthropic/claude-sonnet-4-5')
        ... )

        With additional parameters:

        >>> tbl.add_computed_column(
        ...     response=chat_completions(
        ...         messages,
        ...         model='openai/gpt-4o',
        ...         model_kwargs={'temperature': 0.7, 'max_tokens': 500},
        ...     )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    Env.get().require_package('litellm')

    import litellm

    result = await litellm.acompletion(
        model=model,
        messages=messages,
        drop_params=True,
        **model_kwargs,
    )
    return result.model_dump(mode='json')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

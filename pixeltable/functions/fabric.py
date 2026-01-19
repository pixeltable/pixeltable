"""
Pixeltable UDFs that wrap Azure OpenAI endpoints via Microsoft Fabric.

These functions provide seamless access to Azure OpenAI models within Microsoft Fabric
notebook environments. Authentication and endpoint discovery are handled automatically
using Fabric's built-in service discovery and token utilities.

**Note:** These functions only work within Microsoft Fabric notebook environments.

For more information on Fabric AI services, see:
<https://learn.microsoft.com/en-us/fabric/data-science/ai-services/ai-services-overview>
"""

import datetime
import json
import logging
from typing import Any

import httpx
import numpy as np

import pixeltable as pxt
from pixeltable import env, type_system as ts
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

_logger = logging.getLogger('pixeltable')


def _get_fabric_config() -> tuple[Any, str]:
    """Get Fabric environment configuration and auth header.

    Returns:
        tuple: (fabric_env_config, auth_header)

    Raises:
        ImportError: If Fabric SDK packages are not available.
    """
    try:
        from synapse.ml.fabric.service_discovery import get_fabric_env_config
        from synapse.ml.fabric.token_utils import TokenUtils
    except ImportError as e:
        raise ImportError(
            "Microsoft Fabric SDK packages are required to use Fabric integration. "
            "These packages are only available in Microsoft Fabric notebook environments. "
            "Please ensure you are running in a Fabric environment."
        ) from e

    fabric_env_config = get_fabric_env_config().fabric_env_config
    auth_header = TokenUtils().get_openai_auth_header()
    return fabric_env_config, auth_header


def _is_reasoning_model(model: str) -> bool:
    """Detect if a model is a reasoning model (e.g., gpt-5 family).

    Reasoning models have different parameter requirements:
    - Use max_completion_tokens instead of max_tokens
    - Don't support temperature parameter

    Args:
        model: The model deployment name.

    Returns:
        bool: True if the model is a reasoning model.
    """
    # Future-proof: handles gpt-5, gpt-5-turbo, etc.
    return model.startswith('gpt-5') or 'reasoning' in model.lower()


def _get_header_info(
    headers: httpx.Headers,
) -> tuple[tuple[int, int, datetime.datetime] | None, tuple[int, int, datetime.datetime] | None]:
    """Parse Azure OpenAI rate limit headers.

    Azure OpenAI uses the same header format as OpenAI:
    - x-ratelimit-remaining-requests
    - x-ratelimit-remaining-tokens
    - x-ratelimit-reset-requests
    - x-ratelimit-reset-tokens

    Args:
        headers: Response headers from Azure OpenAI API.

    Returns:
        tuple: (requests_info, tokens_info) where each is (limit, remaining, reset_ts) or None
    """
    def _get_resource_info(resource: str) -> tuple[int, int, datetime.datetime] | None:
        remaining_str = headers.get(f'x-ratelimit-remaining-{resource}')
        if remaining_str is None:
            return None
        remaining = int(remaining_str)

        limit_str = headers.get(f'x-ratelimit-limit-{resource}')
        limit = int(limit_str) if limit_str is not None else None

        reset_str = headers.get(f'x-ratelimit-reset-{resource}')
        if reset_str is not None:
            # Azure OpenAI reset header is in ISO format or seconds
            try:
                reset_ts = datetime.datetime.fromisoformat(reset_str.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fallback: treat as seconds from now
                reset_in_seconds = float(reset_str) if reset_str else 5.0
                reset_ts = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=reset_in_seconds)
        else:
            # Default to 5 seconds if no reset header
            reset_ts = datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(seconds=5.0)

        return (limit, remaining, reset_ts)

    requests_info = _get_resource_info('requests')
    tokens_info = _get_resource_info('tokens')

    if requests_info is None or tokens_info is None:
        _logger.debug(f'_get_header_info(): incomplete rate limit info: {headers}')

    return requests_info, tokens_info


class FabricRateLimitsInfo(env.RateLimitsInfo):
    """Rate limiting information for Azure OpenAI in Fabric.

    Handles rate limit tracking and retry logic for Azure OpenAI API calls
    made through Microsoft Fabric.
    """

    def __init__(self, get_request_resources):
        super().__init__(get_request_resources)
        # Azure OpenAI uses httpx.HTTPStatusError for rate limit errors
        self.retryable_errors = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.HTTPStatusError,  # Covers 429, 500, 503
        )

    def record_exc(self, request_ts: datetime.datetime, exc: Exception) -> None:
        """Record rate limit info from an exception.

        Args:
            request_ts: Timestamp when the request was made.
            exc: The exception that was raised.
        """
        if not isinstance(exc, httpx.HTTPStatusError):
            return

        if not hasattr(exc, 'response') or not hasattr(exc.response, 'headers'):
            return

        requests_info, tokens_info = _get_header_info(exc.response.headers)
        _logger.debug(
            f'record_exc(): request_ts: {request_ts}, requests_info={requests_info} tokens_info={tokens_info}'
        )
        self.record(request_ts=request_ts, requests=requests_info, tokens=tokens_info)
        self.has_exc = True

    def _retry_delay_from_exception(self, exc: Exception) -> float | None:
        """Extract retry delay from retry-after header.

        Args:
            exc: The exception to extract retry delay from.

        Returns:
            float: Retry delay in seconds, or None if not available.
        """
        try:
            retry_after_str = exc.response.headers.get('retry-after')
        except AttributeError:
            return None

        if retry_after_str is not None:
            try:
                return float(retry_after_str)
            except ValueError:
                return None
        return None

    def get_retry_delay(self, exc: Exception, attempt: int) -> float | None:
        """Calculate retry delay for an exception.

        Args:
            exc: The exception that occurred.
            attempt: The current retry attempt number.

        Returns:
            float: Delay in seconds before retry, or None if not retryable.
        """
        if not isinstance(exc, self.retryable_errors):
            return None

        # Check for rate limit errors (429)
        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code != 429:
            # Only retry rate limit errors and server errors (5xx)
            if exc.response.status_code < 500:
                return None

        return self._retry_delay_from_exception(exc) or super().get_retry_delay(exc, attempt)


def _chat_completions_get_request_resources(
    messages: list, model: str, model_kwargs: dict[str, Any] | None = None
) -> dict[str, int]:
    """Estimate request resources for chat completions.

    Args:
        messages: The messages list.
        model: The model name.
        model_kwargs: Additional model parameters.

    Returns:
        dict: Estimated resources with 'requests' and 'tokens' keys.
    """
    # Rough estimate: 4 chars per token
    total_chars = sum(
        len(str(msg.get('content', '')))
        for msg in messages
        if isinstance(msg, dict)
    )
    estimated_tokens = int(total_chars / 4)

    # Add max_tokens or max_completion_tokens for output
    max_output = 4000  # default
    if model_kwargs:
        max_output = model_kwargs.get('max_tokens') or model_kwargs.get('max_completion_tokens', 4000)

    return {
        'requests': 1,
        'tokens': estimated_tokens + max_output
    }


def _embeddings_get_request_resources(
    input: list[str], model: str, model_kwargs: dict[str, Any] | None = None
) -> dict[str, int]:
    """Estimate request resources for embeddings.

    Args:
        input: The input text list.
        model: The model name.
        model_kwargs: Additional model parameters.

    Returns:
        dict: Estimated resources with 'requests' and 'tokens' keys.
    """
    # Rough estimate: 4 chars per token
    total_chars = sum(len(text) for text in input)
    estimated_tokens = int(total_chars / 4)

    return {
        'requests': 1,
        'tokens': estimated_tokens
    }


@pxt.udf
async def chat_completions(
    messages: list[dict],
    *,
    model: str,
    api_version: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> dict:
    """
    Creates a model response for the given chat conversation using Azure OpenAI in Fabric.

    Equivalent to the Azure OpenAI `chat/completions` API endpoint.
    For additional details, see: <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>

    **Automatic authentication:** Authentication is handled automatically in Fabric notebooks using
    token-based authentication. No API keys are required.

    **Supported models in Fabric:**
    - `gpt-5` (reasoning model)
    - `gpt-4.1`
    - `gpt-4.1-mini`

    Request throttling:
    Uses the rate limit-related headers returned by the Azure OpenAI API to throttle requests adaptively,
    based on available request and token capacity. No configuration is necessary.

    __Requirements:__

    - Microsoft Fabric notebook environment
    - `synapse-ml-fabric` package (pre-installed in Fabric)

    Args:
        messages: A list of message dicts with 'role' and 'content' keys, as described in the
            Azure OpenAI API documentation.
        model: The deployment name to use (e.g., 'gpt-5', 'gpt-4.1', 'gpt-4.1-mini').
        api_version: Optional API version override. If not specified, defaults to '2025-04-01-preview'
            for reasoning models (gpt-5) and '2024-02-15-preview' for standard models.
        model_kwargs: Additional keyword args for the Azure OpenAI `chat/completions` API.
            For details on available parameters, see:
            <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>

            **Note:** Reasoning models (gpt-5) use `max_completion_tokens` instead of `max_tokens`
            and do not support the `temperature` parameter.

    Returns:
        A dictionary containing the response and other metadata.

    Examples:
        Add a computed column that applies the model `gpt-4.1` to an existing Pixeltable column
        `tbl.prompt` of the table `tbl`:

        >>> from pixeltable.functions import fabric
        >>> messages = [
        ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
        ...     {'role': 'user', 'content': tbl.prompt}
        ... ]
        >>> tbl.add_computed_column(response=fabric.chat_completions(messages, model='gpt-4.1'))

        Using a reasoning model (gpt-5):

        >>> tbl.add_computed_column(
        ...     reasoning_response=fabric.chat_completions(
        ...         messages,
        ...         model='gpt-5',
        ...         model_kwargs={'max_completion_tokens': 5000}
        ...     )
        ... )
    """
    if model_kwargs is None:
        model_kwargs = {}

    # Get Fabric config and auth
    fabric_env_config, auth_header = _get_fabric_config()

    # Determine API version based on model type if not specified
    if api_version is None:
        api_version = "2025-04-01-preview" if _is_reasoning_model(model) else "2024-02-15-preview"

    # Build URL
    url = (
        f"{fabric_env_config.ml_workload_endpoint}cognitive/openai/openai/deployments/"
        f"{model}/chat/completions?api-version={api_version}"
    )

    # Build payload
    payload = {"messages": messages}

    # Handle reasoning vs standard models
    if _is_reasoning_model(model):
        # Reasoning models use max_completion_tokens, no temperature
        # Extract max_tokens if present and convert to max_completion_tokens
        if 'max_tokens' in model_kwargs:
            payload['max_completion_tokens'] = model_kwargs.pop('max_tokens')
        elif 'max_completion_tokens' in model_kwargs:
            payload['max_completion_tokens'] = model_kwargs.pop('max_completion_tokens')
        else:
            payload['max_completion_tokens'] = 4000

        # Add remaining kwargs (excluding temperature and n)
        for k, v in model_kwargs.items():
            if k not in ('temperature', 'n'):
                payload[k] = v
    else:
        # Standard models support all parameters
        payload.update(model_kwargs)
        payload.setdefault('max_tokens', 4000)
        payload.setdefault('temperature', 0.0)

    # Make request
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        return response.json()


@pxt.udf(batch_size=32)
async def embeddings(
    input: Batch[str],
    *,
    model: str = "text-embedding-ada-002",
    api_version: str = "2024-02-15-preview",
    model_kwargs: dict[str, Any] | None = None,
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Creates an embedding vector representing the input text using Azure OpenAI in Fabric.

    Equivalent to the Azure OpenAI `embeddings` API endpoint.
    For additional details, see: <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>

    **Automatic authentication:** Authentication is handled automatically in Fabric notebooks using
    token-based authentication. No API keys are required.

    **Supported models in Fabric:**
    - `text-embedding-ada-002`
    - `text-embedding-3-small`
    - `text-embedding-3-large`

    Request throttling:
    Uses the rate limit-related headers returned by the Azure OpenAI API to throttle requests adaptively,
    based on available request and token capacity. Batches up to 32 inputs per request for efficiency.

    __Requirements:__

    - Microsoft Fabric notebook environment
    - `synapse-ml-fabric` package (pre-installed in Fabric)

    Args:
        input: The text to embed (automatically batched).
        model: The embedding model deployment name (default: 'text-embedding-ada-002').
        api_version: The API version to use (default: '2024-02-15-preview').
        model_kwargs: Additional keyword args for the Azure OpenAI `embeddings` API.
            For details on available parameters, see:
            <https://learn.microsoft.com/en-us/azure/ai-services/openai/reference>

    Returns:
        An array representing the embedding vector for the input text.

    Examples:
        Add a computed column that applies the model `text-embedding-ada-002` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> from pixeltable.functions import fabric
        >>> tbl.add_computed_column(embed=fabric.embeddings(tbl.text))

        Add an embedding index to an existing column `text`:

        >>> tbl.add_embedding_index('text', embedding=fabric.embeddings.using(model='text-embedding-ada-002'))
    """
    if model_kwargs is None:
        model_kwargs = {}

    _logger.debug(f'embeddings: batch_size={len(input)}')

    # Get Fabric config and auth
    fabric_env_config, auth_header = _get_fabric_config()

    # Build URL
    url = (
        f"{fabric_env_config.ml_workload_endpoint}cognitive/openai/openai/deployments/"
        f"{model}/embeddings?api-version={api_version}"
    )

    # Build payload
    payload = {"input": list(input)}
    payload.update(model_kwargs)

    # Make request
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
        data = response.json()

    # Return embeddings as numpy arrays (same format as OpenAI)
    return [np.array(item["embedding"], dtype=np.float64) for item in data["data"]]


@embeddings.conditional_return_type
def _(model: str = "text-embedding-ada-002", model_kwargs: dict[str, Any] | None = None) -> ts.ArrayType:
    """Determine the return type based on the model."""
    # Known embedding dimensions for common models
    embedding_dimensions = {
        'text-embedding-ada-002': 1536,
        'text-embedding-3-small': 1536,
        'text-embedding-3-large': 3072,
    }

    # Check if dimensions are specified in model_kwargs
    dimensions = None
    if model_kwargs is not None:
        dimensions = model_kwargs.get('dimensions')

    # If not specified, use known dimensions for the model
    if dimensions is None:
        dimensions = embedding_dimensions.get(model, 1536)  # Default to 1536

    return ts.ArrayType((dimensions,), dtype=ts.FloatType(), nullable=False)


__all__ = local_public_names(__name__)

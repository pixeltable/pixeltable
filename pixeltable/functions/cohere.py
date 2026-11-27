"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap [Cohere](https://cohere.com/)
APIs for embeddings and reranking. Install the `cohere` SDK first. Specify the API key with `COHERE_API_KEY`
environment variable, or as `api_key` in the `cohere` section of the Pixeltable config file.
"""

import json
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import httpx
import numpy as np

import pixeltable as pxt
from pixeltable import type_system as ts
from pixeltable.env import Env, register_client
from pixeltable.func import Batch
from pixeltable.runtime import get_runtime
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from cohere import AsyncClientV2

_MAX_DETAIL_LEN = 500

# Default embedding dimensions for Cohere models
_embedding_dimensions_cache: dict[str, int] = {
    'embed-v5.0-pro': 2048,
    'embed-v5.0-fast': 2048,
    'embed-v4.0': 1536,
    'embed-english-v3.0': 1024,
    'embed-multilingual-v3.0': 1024,
    'embed-english-light-v3.0': 384,
    'embed-multilingual-light-v3.0': 384,
}


def _error_detail(body: str) -> str:
    """Returns the error message of a failed response, as the API reports it."""
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and isinstance(parsed.get('message'), str):
        return parsed['message'][:_MAX_DETAIL_LEN]
    return body[:_MAX_DETAIL_LEN]


def _parse_retry_after(value: str | None) -> float | None:
    """Returns the delay in seconds given by a `Retry-After` header, if it holds a number of seconds."""
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        return None
    return delay if delay >= 0 else None


def _status_error(endpoint: str, status: int, retry_after: str | None, body: str) -> pxt.ExternalServiceError:
    code: pxt.ErrorCode
    if status in (401, 402, 403, 498):
        # 498 is Cohere's status for an invalid API key, 402 for a billing problem with the account
        code = pxt.ErrorCode.PROVIDER_AUTH_ERROR
    elif status == 429:
        code = pxt.ErrorCode.RATE_LIMITED
    elif status == 503:
        code = pxt.ErrorCode.PROVIDER_OVERLOADED
    elif status >= 500:
        code = pxt.ErrorCode.PROVIDER_ERROR
    else:
        code = pxt.ErrorCode.PROVIDER_BAD_REQUEST
    return pxt.ExternalServiceError(
        code,
        f'Cohere request to {endpoint} failed with status {status}: {_error_detail(body)}',
        retry_after=_parse_retry_after(retry_after) if status in (429, 503) else None,
        provider='cohere',
        status_code=status,
    )


class _CohereClient:
    """Owns the SDK transport and adapts provider failures to Pixeltable errors."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._sdk: AsyncClientV2 | None = None
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    def _get_sdk(self) -> 'AsyncClientV2':
        if self._sdk is None:
            Env.get().require_package('cohere', [7, 3, 2])
            from cohere import AsyncClientV2

            self._http_client = self._get_http_client()
            self._sdk = AsyncClientV2(api_key=self._api_key, httpx_client=self._http_client, max_retries=0)
        return self._sdk

    async def close(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._sdk = None

    async def _post(self, endpoint: str, *, payload: dict[str, Any]) -> dict[str, Any]:
        sdk = self._get_sdk()
        from cohere.core.api_error import ApiError

        try:
            if endpoint == '/v2/embed':
                result = (await sdk.embed(**payload)).model_dump(by_alias=True, exclude_none=True)
            else:
                result = (await sdk.rerank(**payload)).model_dump(by_alias=True, exclude_none=True)
        except ApiError as exc:
            if exc.status_code is not None and exc.status_code < 400:
                raise pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                    'Cohere returned a response that is not a JSON object',
                    provider='cohere',
                ) from exc
            body = exc.body if isinstance(exc.body, str) else json.dumps(exc.body)
            raise _status_error(
                endpoint, exc.status_code or 500, httpx.Headers(exc.headers or {}).get('Retry-After'), body
            ) from exc
        except (TypeError, ValueError) as exc:
            raise pxt.ExternalServiceError(
                pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
                'Cohere returned a response that is not a JSON object',
                provider='cohere',
            ) from exc
        return result


@register_client('cohere', credential_param='api_key')
def _(api_key: str) -> _CohereClient:
    return _CohereClient(api_key=api_key)


def _client() -> _CohereClient:
    return get_runtime().get_client('cohere')


@pxt.udf(batch_size=96, resource_pool='request-rate:cohere')
async def embeddings(
    input: Batch[str],
    *,
    model: str,
    input_type: Literal['search_document', 'search_query', 'classification', 'clustering'],
    truncate: Literal['NONE', 'START', 'END'] | None = None,
    output_dimension: int | None = None,
) -> Batch[pxt.Array[(None,), np.float32]]:
    """
    Creates embedding vectors for the input text using Cohere embedding models.

    Equivalent to the Cohere `embed` API endpoint (v2).
    For additional details, see: <https://docs.cohere.com/reference/embed>

    Request throttling:
    Applies the rate limit set in the config (section `cohere`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        input: The text to embed.
        model: The Cohere embedding model to use, such as `embed-v4.0` or `embed-english-v3.0`. See available
            models at <https://docs.cohere.com/docs/cohere-embed>.
        input_type: The type of the input, which the model uses to optimize the embedding. Options:

            - `search_document`: For documents stored in a vector database for search
            - `search_query`: For search queries run against such documents
            - `classification`: For text classifiers
            - `clustering`: For clustering

        truncate: How to handle inputs longer than the model's maximum context length. Options: `NONE` (return an
            error), `START` (drop the start of the input) and `END` (drop the end of the input, the API's default).
        output_dimension: Output embedding dimensions (optional). Only supported by `embed-v4.0` and newer models;
            for example, `embed-v4.0` accepts `256`, `512`, `1024` or `1536`. If not specified, uses the model's
            default dimension.

    Returns:
        An array representing the embedding of `input`.

    Examples:
        Add a computed column that applies `embed-v4.0` to an existing column:

        >>> tbl.add_computed_column(
        ...     embed=cohere.embeddings(
        ...         tbl.text, model='embed-v4.0', input_type='search_document'
        ...     )
        ... )

        Cohere embeds documents and queries differently. To search the documents, add an embedding index on the
        column of document embeddings, with a query-side embedding function:

        >>> tbl.add_embedding_index(
        ...     'embed',
        ...     string_embed=cohere.embeddings.using(
        ...         model='embed-v4.0', input_type='search_query'
        ...     ),
        ... )
        >>> sim = tbl.embed.similarity(string='How do I reset my password?')
        >>> tbl.order_by(sim, asc=False).limit(5).select(tbl.text, sim).collect()
    """
    cl = _client()

    payload: dict[str, Any] = {'model': model, 'input_type': input_type, 'texts': input, 'embedding_types': ['float']}
    if truncate is not None:
        payload['truncate'] = truncate
    if output_dimension is not None:
        payload['output_dimension'] = output_dimension

    result = await cl._post('/v2/embed', payload=payload)

    # The vectors are returned in the order of the input texts, grouped by embedding type
    embeddings_by_type = result.get('embeddings')
    vectors = embeddings_by_type.get('float') if isinstance(embeddings_by_type, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(input):
        num_vectors = len(vectors) if isinstance(vectors, list) else 0
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'Cohere returned {num_vectors} float embeddings for {len(input)} input texts',
            provider='cohere',
        )
    return [np.array(vector, dtype=np.float32) for vector in vectors]


@embeddings.conditional_return_type
def _(model: str, output_dimension: int | None) -> ts.ArrayType:
    # If output_dimension is explicitly specified, use it
    if output_dimension is not None:
        return ts.ArrayType((output_dimension,), dtype=np.dtype(np.float32), nullable=False)
    # Otherwise, look up the default for this model
    dim = _embedding_dimensions_cache.get(model)
    if dim is None:
        return ts.ArrayType((None,), dtype=np.dtype(np.float32), nullable=False)
    return ts.ArrayType((dim,), dtype=np.dtype(np.float32), nullable=False)


class CohereRerankResult(TypedDict):
    index: int
    relevance_score: float


class CohereRerankResponse(TypedDict):
    results: list[CohereRerankResult]
    meta: dict


@pxt.udf(resource_pool='request-rate:cohere')
async def rerank(
    query: str, documents: list[str], *, model: str, top_n: int | None = None, max_tokens_per_doc: int | None = None
) -> CohereRerankResponse:
    """
    Reranks documents based on their relevance to a query using Cohere reranker models.

    Equivalent to the Cohere `rerank` API endpoint (v2).
    For additional details, see: <https://docs.cohere.com/reference/rerank>

    Request throttling:
    Applies the rate limit set in the config (section `cohere`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    Args:
        query: The query string to rank documents against.
        documents: The list of documents to rerank.
        model: The Cohere reranker model to use, such as `rerank-v3.5`. See available models at
            <https://docs.cohere.com/docs/rerank>.
        top_n: Number of top results to return. If not specified, returns all documents.
        max_tokens_per_doc: Documents longer than this many tokens are truncated. The API's default is 4096.

    Returns:
        A dictionary containing:

        - `results`: List of reranking results with `index` (the position of the document in `documents`) and
            `relevance_score`, ordered by decreasing relevance
        - `meta`: Metadata about the request, such as billed units

    Examples:
        Rerank search results for better relevance:

        >>> tbl.add_computed_column(
        ...     reranked=cohere.rerank(
        ...         tbl.query, tbl.candidate_docs, model='rerank-v3.5', top_n=5
        ...     )
        ... )
    """
    cl = _client()

    payload: dict[str, Any] = {'model': model, 'query': query, 'documents': documents}
    if top_n is not None:
        payload['top_n'] = top_n
    if max_tokens_per_doc is not None:
        payload['max_tokens_per_doc'] = max_tokens_per_doc

    result = await cl._post('/v2/rerank', payload=payload)

    results = result.get('results')
    if not isinstance(results, list) or not all(
        isinstance(r, dict) and 'index' in r and 'relevance_score' in r for r in results
    ):
        raise pxt.ExternalServiceError(
            pxt.ErrorCode.PROVIDER_UNEXPECTED_RESPONSE,
            f'Cohere returned an unexpected rerank response: {str(result)[:_MAX_DETAIL_LEN]}',
            provider='cohere',
        )
    formatted_results: list[CohereRerankResult] = [
        {'index': r['index'], 'relevance_score': r['relevance_score']} for r in results
    ]
    return {'results': formatted_results, 'meta': result.get('meta', {})}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

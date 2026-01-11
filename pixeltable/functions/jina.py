"""
Pixeltable [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) that wrap [Jina AI](https://jina.ai/) APIs
for embeddings and reranking. In order to use them, the API key must be specified either with `JINA_API_KEY`
environment variable, or as `api_key` in the `jina` section of the Pixeltable config file.
"""

import asyncio
import atexit
import logging
import re
from typing import Any, Literal

import aiohttp
import numpy as np

import pixeltable as pxt
from pixeltable import type_system as ts
from pixeltable.env import Env, register_client
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

_logger = logging.getLogger('pixeltable')

# Default embedding dimensions for Jina models
_embedding_dimensions_cache: dict[str, int] = {
    'jina-embeddings-v4': 2048,
    'jina-clip-v2': 1024,
    'jina-embeddings-v3': 1024,
    'jina-clip-v1': 768,
    'jina-embeddings-v2-base-es': 768,
    'jina-embeddings-v2-base-code': 768,
    'jina-embeddings-v2-base-de': 768,
    'jina-embeddings-v2-base-zh': 768,
    'jina-embeddings-v2-base-en': 768,
    'jina-code-embeddings-0.5b': 896,
    'jina-code-embeddings-1.5b': 1536,
}


class JinaRateLimitedError(Exception):
    pass


class JinaUnexpectedError(Exception):
    pass


class _JinaClient:
    """
    Client for interacting with the Jina AI API. Maintains a long-lived HTTP session to the service.
    """

    api_key: str
    session: aiohttp.ClientSession

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = Env.get().event_loop.run_until_complete(self._start_session())
        atexit.register(lambda: asyncio.run(self.session.close()))

    async def _start_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(base_url='https://api.jina.ai')

    async def _post(self, endpoint: str, *, payload: dict) -> dict:
        request_headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

        async with self.session.post(endpoint, json=payload, headers=request_headers) as resp:
            match resp.status:
                case 200:
                    data = await resp.json()
                    return data
                case 429:
                    retry_after_seconds = None
                    retry_after_header = resp.headers.get('Retry-After')
                    if retry_after_header is not None and re.fullmatch(r'\d{1,2}', retry_after_header):
                        retry_after_seconds = int(retry_after_header)
                    _logger.info(f'Jina request failed due to rate limiting, retry after: {retry_after_header}')
                    raise JinaRateLimitedError(
                        f'Jina request failed due to rate limiting (429). retry-after:{retry_after_seconds}'
                    )
                case _:
                    error_text = await resp.text()
                    _logger.info(f'Jina request failed with status code {resp.status}: {error_text}')
                    raise JinaUnexpectedError(f'Jina API error (status {resp.status}): {error_text}')


@register_client('jina')
def _(api_key: str) -> _JinaClient:
    return _JinaClient(api_key=api_key)


def _client() -> _JinaClient:
    return Env.get().get_client('jina')


@pxt.udf(batch_size=128, resource_pool='request-rate:jina')
async def embeddings(
    input: Batch[str],
    *,
    model: str,
    task: Literal['retrieval.query', 'retrieval.passage', 'separation', 'classification', 'text-matching']
    | None = None,
    dimensions: int | None = None,
    late_chunking: bool | None = None,
) -> Batch[pxt.Array[(None,), np.float32]]:
    """
    Creates embedding vectors for the input text using Jina AI embedding models.

    Equivalent to the Jina AI embeddings API endpoint.
    For additional details, see: <https://jina.ai/embeddings/>

    Request throttling:
    Applies the rate limit set in the config (section `jina`, key `rate_limit`). If no rate
    limit is configured, uses a default of 500 RPM.

    Args:
        input: The text to embed.
        model: The Jina embedding model to use. See available models at
            <https://jina.ai/embeddings/>.
        task: Task-specific embedding optimization. Options:

            - `retrieval.query`: For search queries
            - `retrieval.passage`: For documents/passages to be searched
            - `separation`: For clustering/separation tasks
            - `classification`: For classification tasks
            - `text-matching`: For semantic similarity

        dimensions: Output embedding dimensions (optional). If not specified, uses
            the model's default dimension.
        late_chunking: Enable late chunking for long documents.

    Returns:
        An array representing the embedding of `input`.

    Examples:
        Add a computed column that applies jina-embeddings-v3 to an existing column:

        >>> tbl.add_computed_column(
        ...     embed=jina.embeddings(tbl.text, model='jina-embeddings-v3', task='retrieval.passage')
        ... )

        Add an embedding index:

        >>> tbl.add_embedding_index('text', string_embed=jina.embeddings.using(model='jina-embeddings-v3'))
    """
    cl = _client()

    payload: dict[str, Any] = {'model': model, 'input': input}

    if task is not None:
        payload['task'] = task
    if dimensions is not None:
        payload['dimensions'] = dimensions
    if late_chunking is not None:
        payload['late_chunking'] = late_chunking

    result = await cl._post('/v1/embeddings', payload=payload)

    # Extract embeddings from response - they come as a list of objects with 'embedding' field
    embeddings_list = result.get('data', [])
    # Sort by index to ensure correct order
    embeddings_list.sort(key=lambda x: x.get('index', 0))
    return [np.array(item['embedding'], dtype=np.float32) for item in embeddings_list]


@embeddings.conditional_return_type
def _(model: str, dimensions: int | None) -> ts.ArrayType:
    # If dimensions is explicitly specified, use it
    if dimensions is not None:
        return ts.ArrayType((dimensions,), dtype=ts.FloatType(), nullable=False)
    # Otherwise, look up the default for this model
    dim = _embedding_dimensions_cache.get(model)
    if dim is None:
        return ts.ArrayType((None,), dtype=ts.FloatType(), nullable=False)
    return ts.ArrayType((dim,), dtype=ts.FloatType(), nullable=False)


@pxt.udf(resource_pool='request-rate:jina')
async def rerank(
    query: str, documents: list[str], *, model: str, top_n: int | None = None, return_documents: bool | None = None
) -> dict:
    """
    Reranks documents based on their relevance to a query using Jina AI reranker models.

    Equivalent to the Jina AI rerank API endpoint.
    For additional details, see: <https://jina.ai/reranker/>

    Request throttling:
    Applies the rate limit set in the config (section `jina`, key `rate_limit`). If no rate
    limit is configured, uses a default of 500 RPM.

    Args:
        query: The query string to rank documents against.
        documents: The list of documents to rerank.
        model: The Jina reranker model to use. See available models at
            <https://jina.ai/reranker/>.
        top_n: Number of top results to return. If not specified, returns all documents.
        return_documents: Whether to include the original document text in results.

    Returns:
        A dictionary containing:
        - `results`: List of reranking results with `index` and `relevance_score`
            (and `document` if `return_documents=True`)
        - `usage`: Token usage information

    Examples:
        Rerank search results for better relevance:

        >>> tbl.add_computed_column(
        ...     reranked=jina.rerank(
        ...         tbl.query,
        ...         tbl.candidate_docs,
        ...         model='jina-reranker-v2-base-multilingual',
        ...         top_n=5
        ...     )
        ... )
    """
    cl = _client()

    payload: dict[str, Any] = {'model': model, 'query': query, 'documents': documents}

    if top_n is not None:
        payload['top_n'] = top_n
    if return_documents is not None:
        payload['return_documents'] = return_documents

    result = await cl._post('/v1/rerank', payload=payload)

    # Format the response
    results_list = result.get('results', [])
    formatted_results = []
    for r in results_list:
        item = {'index': r.get('index'), 'relevance_score': r.get('relevance_score')}
        if return_documents is True and 'document' in r:
            doc = r['document']
            # Handle both string and dict formats from the API
            item['document'] = doc.get('text', doc) if isinstance(doc, dict) else doc
        formatted_results.append(item)

    return {'results': formatted_results, 'usage': result.get('usage', {})}


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

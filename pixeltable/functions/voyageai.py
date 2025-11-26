"""
Pixeltable UDFs
that wrap various endpoints from the Voyage AI API. In order to use them, you must
first `pip install voyageai` and configure your Voyage AI credentials, as described in
the [Working with Voyage AI](https://docs.pixeltable.com/notebooks/integrations/working-with-voyageai) tutorial.
"""

from typing import TYPE_CHECKING, Literal

import numpy as np

import pixeltable as pxt
from pixeltable import env
from pixeltable.func import Batch
from pixeltable.utils.code import local_public_names

if TYPE_CHECKING:
    from voyageai import AsyncClient


@env.register_client('voyageai')
def _(api_key: str) -> 'AsyncClient':
    from voyageai import AsyncClient

    return AsyncClient(api_key=api_key)


def _voyageai_client() -> 'AsyncClient':
    return env.Env.get().get_client('voyageai')


@pxt.udf(batch_size=128, resource_pool='request-rate:voyageai')
async def embeddings(
    input: Batch[str],
    *,
    model: str,
    input_type: Literal['query', 'document'] | None = None,
    truncation: bool | None = None,
    output_dimension: int | None = None,
    output_dtype: Literal['float', 'int8', 'uint8', 'binary', 'ubinary'] | None = None,
) -> Batch[pxt.Array[(None,), pxt.Float]]:
    """
    Creates an embedding vector representing the input text.

    Equivalent to the Voyage AI `embeddings` API endpoint.
    For additional details, see: <https://docs.voyageai.com/docs/embeddings>

    Request throttling:
    Applies the rate limit set in the config (section `voyageai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install voyageai`

    Args:
        input: The text to embed.
        model: The model to use for the embedding. Recommended options: `voyage-3-large`, `voyage-3.5`,
            `voyage-3.5-lite`, `voyage-code-3`, `voyage-finance-2`, `voyage-law-2`.
        input_type: Type of the input text. Options: `None`, `query`, `document`.
            When `input_type` is `None`, the embedding model directly converts the inputs into numerical vectors.
            For retrieval/search purposes, we recommend setting this to `query` or `document` as appropriate.
        truncation: Whether to truncate the input texts to fit within the context length. Defaults to `True`.
        output_dimension: The number of dimensions for resulting output embeddings.
            Most models only support a single default dimension. Models `voyage-3-large`, `voyage-3.5`,
            `voyage-3.5-lite`, and `voyage-code-3` support: 256, 512, 1024 (default), and 2048.
        output_dtype: The data type for the embeddings to be returned. Options: `float`, `int8`, `uint8`,
            `binary`, `ubinary`. Only `float` is currently supported in Pixeltable.

    Returns:
        An array representing the application of the given embedding to `input`.

    Examples:
        Add a computed column that applies the model `voyage-3.5` to an existing
        Pixeltable column `tbl.text` of the table `tbl`:

        >>> tbl.add_computed_column(embed=embeddings(tbl.text, model='voyage-3.5', input_type='document'))

        Add an embedding index to an existing column `text`, using the model `voyage-3.5`:

        >>> tbl.add_embedding_index('text', text_embed=embeddings.using(model='voyage-3.5'))
    """
    cl = _voyageai_client()

    # Build kwargs for the API call
    kwargs = {}
    if input_type is not None:
        kwargs['input_type'] = input_type
    if truncation is not None:
        kwargs['truncation'] = truncation
    if output_dimension is not None:
        kwargs['output_dimension'] = output_dimension
    if output_dtype is not None:
        kwargs['output_dtype'] = output_dtype

    result = await cl.embed(texts=input, model=model, **kwargs)
    return [np.array(emb, dtype=np.float64) for emb in result.embeddings]


@pxt.udf(resource_pool='request-rate:voyageai')
async def rerank(
    query: str, documents: list[str], *, model: str, top_k: int | None = None, truncation: bool = True
) -> dict:
    """
    Reranks documents based on their relevance to a query.

    Equivalent to the Voyage AI `rerank` API endpoint.
    For additional details, see: <https://docs.voyageai.com/docs/reranker>

    Request throttling:
    Applies the rate limit set in the config (section `voyageai`, key `rate_limit`). If no rate
    limit is configured, uses a default of 600 RPM.

    __Requirements:__

    - `pip install voyageai`

    Args:
        query: The query as a string.
        documents: The documents to be reranked as a list of strings.
        model: The model to use for reranking. Recommended options: `rerank-2.5`, `rerank-2.5-lite`.
        top_k: The number of most relevant documents to return. If not specified, all documents
            will be reranked and returned.
        truncation: Whether to truncate the input to satisfy context length limits. Defaults to `True`.

    Returns:
        A dictionary containing:
        - `results`: List of reranking results with `index`, `document`, and `relevance_score`
        - `total_tokens`: The total number of tokens used

    Examples:
        Rerank a list of documents based on relevance to a query:

        >>> tbl = pxt.create_table('docs', {'query': pxt.String, 'docs': pxt.Json})
        >>> tbl.add_computed_column(
        ...     reranked=rerank(
        ...         tbl.query,
        ...         tbl.docs,
        ...         model='rerank-2.5',
        ...         top_k=3
        ...     )
        ... )
    """
    cl = _voyageai_client()

    result = await cl.rerank(query=query, documents=documents, model=model, top_k=top_k, truncation=truncation)

    # Convert the result to a dictionary format
    return {
        'results': [
            {'index': r.index, 'document': r.document, 'relevance_score': r.relevance_score} for r in result.results
        ],
        'total_tokens': result.total_tokens,
    }


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

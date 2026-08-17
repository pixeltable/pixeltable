"""A chunked view with an embedding index, and query routes over it.

The embedding comes from the test utilities rather than the application file: an index stores the symbol of
its embedding function, and a function defined here would not resolve on a later load. See
TODO(udf-in-app-file) in basic.py.
"""

# ruff: noqa: F821  # a model body refers to its own columns, and an iterator's, by bare name
# ruff: noqa: RUF012  # __indexes__ is the declaration syntax, not a mutable class attribute

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import EmbeddingIndex
from pixeltable.serving import FastAPIRouter
from tests.utils import dummy_embedding

TableModel = pxt.model_base()


class Articles(TableModel, name='articles'):
    article_id = pxt.Column(type=pxt.Int, primary_key=True)
    body: pxt.String
    word_count = pxtf.string.len(body)
    __indexes__ = [pxt.BtreeIndex(article_id)]


class Chunks(
    TableModel, name='chunks', base=Articles, iterator=pxtf.string.string_splitter(Articles.body, separators='sentence')
):
    """One row per sentence of an article, with an index over the text the iterator produces."""

    __indexes__ = [EmbeddingIndex(text, embedding=dummy_embedding.using(n=32), name='chunk_ix')]  # type: ignore[name-defined]


# TODO(udf-in-app-file): restore this query and the two routes below. A route records the symbol of the
# function it calls, and the loader gives the application file a throwaway module name, so the recorded
# symbol differs on every load: the diff reports 'query changed' every time, the service never reaches
# up_to_date, and `update` refuses it as destructive. Same cause as the udf in basic.py.
#
# @pxt.query
# def similar_chunks(needle: str, limit: int = 3) -> pxt.Query:
#     """The chunks closest to a search string, which is what an embedding index is for."""
#     sim = Chunks.text.similarity(string=needle)
#     return Chunks.order_by(sim, asc=False).limit(limit).select(Chunks.text)


search = FastAPIRouter(name='search')
search.add_insert_route(Articles, path='/articles', inputs=[Articles.article_id, Articles.body])  # type: ignore[arg-type]
# search.add_query_route(path='/similar', query=similar_chunks)
# search.add_query_route(path='/similar-one', query=similar_chunks, one_row=True, method='get')

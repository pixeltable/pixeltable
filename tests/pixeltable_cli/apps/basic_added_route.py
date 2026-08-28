"""basic.py plus one route: the additive drift a diff reports and update applies.

A corpus file the schema and service CLI tests share: `pxt schema update` creates what the models declare,
`pxt service update` serves what the router declares over the same tables.
"""

# ruff: noqa: F821  # a model body refers to its own columns by bare name

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


@pxt.udf
def excerpt(text: str, n: int = 12) -> str:
    """A udf, so that a computed column is not only an expression over other columns."""
    return text if len(text) <= n else f'{text[:n]}...'


class Docs(TableModel, name='docs'):
    doc_id = pxt.Column(type=pxt.Int, primary_key=True)
    title: pxt.String
    body: pxt.String | None
    rating: pxt.Float | None
    published: pxt.Bool
    posted_at: pxt.Timestamp | None
    tags: pxt.Json | None
    title_upper = pxtf.string.upper(title)
    summary = excerpt(title)
    unstored = pxt.Column(value=pxtf.string.lower(title), stored=False)


class Published(TableModel, name='published', base=Docs.where(Docs.published)):
    """A view of the rows that satisfy a filter."""

    headline = Docs.title_upper + '!'


class Titles(TableModel, name='titles', base=Docs.select(Docs.doc_id, t=Docs.title)):
    """A view that projects its base rather than inheriting every column."""

    shouted = t + '!'  # type: ignore[name-defined]  # t is the select() alias, referenceable in the body


ingest = FastAPIRouter(name='ingest')
ingest.add_insert_route(
    Docs,
    path='/docs',
    # every column the table requires has to be an input; the nullable ones may be left out
    inputs=[Docs.doc_id, Docs.title, Docs.body, Docs.published],  # type: ignore[arg-type]
    outputs=[Docs.title_upper, Docs.summary],  # type: ignore[arg-type]
)
# a compute route builds a row without storing it, so it too has to be given every required column
ingest.add_compute_route(
    Docs,
    path='/preview',
    inputs=[Docs.doc_id, Docs.title, Docs.published],  # type: ignore[arg-type]
    outputs=[Docs.summary],  # type: ignore[arg-type]
)
# the row is identified by its primary key, which the request carries but which is not an input
ingest.add_update_route(
    Docs,
    path='/docs/update',
    inputs=[Docs.title],
    outputs=[Docs.title_upper],  # type: ignore[arg-type]
)
ingest.add_delete_route(Docs, path='/docs/delete')
ingest.add_compute_route(
    Docs,
    path='/shout',
    inputs=[Docs.doc_id, Docs.title, Docs.published],  # type: ignore[arg-type]
    outputs=[Docs.title_upper],  # type: ignore[arg-type]
)

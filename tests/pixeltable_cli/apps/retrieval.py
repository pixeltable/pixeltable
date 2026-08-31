"""A computed column that calls a query udf."""

import pixeltable as pxt

TableModel = pxt.model_base()


class Docs(TableModel, name='docs'):
    doc_id = pxt.Column(type=pxt.Int, primary_key=True)
    title: pxt.String


# mypy sees the model's columns as their declared Python types, not as expressions
# mypy: disable-error-code="arg-type, operator"


@pxt.query
def titles_after(cutoff: int) -> pxt.Query:
    """The titles of the documents past a cutoff, ordered so the result is stable."""
    return Docs.where(Docs.doc_id > cutoff).order_by(Docs.doc_id).select(Docs.title)


class Probe(TableModel, name='probe'):
    cutoff = pxt.Column(type=pxt.Int, primary_key=True)
    matches = titles_after(cutoff)

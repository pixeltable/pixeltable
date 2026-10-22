"""An application object that includes a router under a prefix, alongside a route written by hand.

The prefix is what distinguishes a router's paths in the application from the paths the application serves
itself: `/v1/notes` is the router's, `/hand-written` is the application's own.
"""

# ruff: noqa: F821  # a model body refers to its own columns by bare name

from __future__ import annotations

import fastapi

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Notes(TableModel, name='notes'):
    note_id = pxt.Column(type=pxt.Int, primary_key=True)
    text: pxt.String
    text_upper = pxtf.string.upper(text)


plain = FastAPIRouter(name='plain')
plain.add_insert_route(Notes, path='/notes', inputs=[Notes.note_id, Notes.text])  # type: ignore[arg-type]

app = fastapi.FastAPI()
app.include_router(plain, prefix='/v1')


@app.get('/hand-written')
def hand_written() -> dict[str, str]:
    return {'written': 'by hand'}

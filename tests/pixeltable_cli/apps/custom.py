"""An application object the file supplies itself, alongside a router.

Pixeltable declared none of the application's routes, so it can neither compare nor serve them; the router
in the same file is compared as usual. Serving this case is deferred -- see the plan's custom-app entry.
"""

# ruff: noqa: F821  # a model body refers to its own columns by bare name

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

# the application the file supplies: the router above is part of it, alongside routes written by hand
app = fastapi.FastAPI()
app.include_router(plain)


@app.get('/hand-written')
def hand_written() -> dict[str, str]:
    return {'written': 'by hand'}

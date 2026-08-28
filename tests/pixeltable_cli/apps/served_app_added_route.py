"""served_app.py with one more route, so that a diff reports an addition.

`pxt schema update` creates what the model declares; `pxt service update` serves this application as it is,
so its handlers reach the tables through the Pixeltable API like any other program.
"""

# ruff: noqa: F821  # a model body refers to its own columns by bare name

from __future__ import annotations

import fastapi

import pixeltable as pxt
import pixeltable.functions as pxtf

TableModel = pxt.model_base()


class Notes(TableModel, name='notes'):
    note_id = pxt.Column(type=pxt.Int, primary_key=True)
    text: pxt.String
    text_upper = pxtf.string.upper(text)


scribe = fastapi.FastAPI(title='scribe')


# the models this file declares are bound when the service starts, so a handler reaches them by name
@scribe.post('/notes')
def add_note(note_id: int, text: str) -> dict[str, int]:
    status = Notes.insert([{'note_id': note_id, 'text': text}])
    return {'rows': status.num_rows}


@scribe.get('/notes/count')
def count_notes() -> dict[str, int]:
    return {'count': Notes.count()}


@scribe.get('/notes/upper')
def upper_notes() -> dict[str, list[str]]:
    return {'upper': [row['text_upper'] for row in Notes.select(Notes.text_upper).collect()]}

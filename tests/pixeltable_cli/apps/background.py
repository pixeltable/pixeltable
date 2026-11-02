"""A background insert route over a computed column that takes a while."""

# ruff: noqa: F821  # a model body refers to its own columns by bare name

import pixeltable as pxt
from apps.udfs import sleep
from pixeltable.serving import FastAPIRouter

TableModel = pxt.model_base()


class Naps(TableModel, name='naps'):
    nap_id = pxt.Column(type=pxt.Int, primary_key=True)
    secs: pxt.Float
    slept = sleep(secs)


jobs = FastAPIRouter(name='jobs')
jobs.add_insert_route(
    Naps,
    path='/naps',
    inputs=[Naps.nap_id, Naps.secs],  # type: ignore[arg-type]
    outputs=[Naps.nap_id, Naps.slept],
    background=True,
)

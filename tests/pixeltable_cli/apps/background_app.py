"""A custom app with two routers of background routes under one prefix, behind the app's own header check."""

from typing import Annotated

import fastapi

from apps.background import Naps, jobs
from pixeltable.serving import FastAPIRouter

# a second router with a background route, whose name gives it a job route of its own under the shared prefix
more_jobs = FastAPIRouter(name='more-jobs')
more_jobs.add_insert_route(
    Naps, path='/more-naps', inputs=[Naps.nap_id, Naps.secs], outputs=[Naps.nap_id, Naps.slept], background=True
)


def check_tenant(x_tenant: Annotated[str, fastapi.Header()]) -> None:
    if x_tenant != 'acme':
        raise fastapi.HTTPException(status_code=403, detail='unknown tenant')


app = fastapi.FastAPI()
app.include_router(jobs, prefix='/api', dependencies=[fastapi.Depends(check_tenant)])
app.include_router(more_jobs, prefix='/api', dependencies=[fastapi.Depends(check_tenant)])

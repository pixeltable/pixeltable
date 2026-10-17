"""A custom app that serves the background route of background.py under a prefix, behind its own header check."""

from typing import Annotated

import fastapi

from apps.background import jobs


def check_tenant(x_tenant: Annotated[str, fastapi.Header()]) -> None:
    if x_tenant != 'acme':
        raise fastapi.HTTPException(status_code=403, detail='unknown tenant')


app = fastapi.FastAPI()
app.include_router(jobs, prefix='/api', dependencies=[fastapi.Depends(check_tenant)])

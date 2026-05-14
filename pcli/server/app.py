from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from pixeltable import exceptions as excs

from .routes import (
    columns,
    count,
    describe,
    drop,
    env,
    errors,
    get,
    health,
    history,
    idxs,
    ls,
    move,
    revert,
    rows,
    status,
)


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    for r in (health, ls, describe, errors, history, columns, idxs, rows, status, env, count, get, drop, move, revert):
        app.include_router(r.router)

    @app.exception_handler(excs.Error)
    def pxt_error_handler(_request: Request, exc: excs.Error) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status, content={'detail': str(exc), 'error_code': exc.error_code.name}
        )

    return app

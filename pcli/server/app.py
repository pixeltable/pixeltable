from fastapi import FastAPI

from .routes import (
    columns, count, describe, drop, env, errors, get, health, history, idxs, ls, move, revert, rows, status,
)


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    for r in (
        health, ls, describe, errors, history, columns, idxs, rows, status, env, count, get,
        drop, move, revert,
    ):
        app.include_router(r.router)
    return app

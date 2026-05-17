from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pcli._paths import redact_home_in_text
from pixeltable import exceptions as excs

from .routes import router


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(router)

    @app.exception_handler(excs.Error)
    def pxt_error_handler(_request: Request, exc: excs.Error) -> JSONResponse:
        # pxt error messages can embed absolute filesystem paths (e.g. media-file open
        # failures); redact before returning so user-facing CLI output doesn't leak the
        # operator's $PIXELTABLE_HOME.
        return JSONResponse(
            status_code=exc.http_status,
            content={'detail': redact_home_in_text(str(exc)), 'error_code': exc.error_code.name},
        )

    @app.exception_handler(RequestValidationError)
    def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Strip pydantic's structured envelope down to a single human line for CLI use.
        msgs = [str(e.get('msg', '')).removeprefix('Value error, ') for e in exc.errors()]
        detail = '; '.join(msgs) if len(msgs) > 0 else 'invalid request'
        return JSONResponse(status_code=400, content={'detail': redact_home_in_text(detail)})

    return app

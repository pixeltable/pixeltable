from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pixeltable import exceptions as excs

from . import state
from .routes import router

# Origins for which the daemon answers CORS preflight: the SPA's Vite dev server (5173)
# and the legacy 8080 origin some demo setups still use.
_DEV_ORIGINS = ['http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080']

# SPA bundle directory: the Vite build drops here. The mount stays attached to the app for
# the daemon's whole lifetime; the 404/200 split when the dashboard is disabled is enforced
# by a per-request gate, not by toggling the mount.
_STATIC_DIR = Path(__file__).parent / 'static'


def create_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_DEV_ORIGINS,
        allow_credentials=False,
        allow_methods=['GET', 'POST', 'OPTIONS'],
        allow_headers=['Content-Type'],
    )
    app.include_router(router)

    @app.exception_handler(excs.Error)
    def pxt_error_handler(_request: Request, exc: excs.Error) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status, content={'detail': str(exc), 'error_code': exc.error_code.name}
        )

    @app.exception_handler(RequestValidationError)
    def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        # Strip pydantic's structured envelope down to a single human line for CLI use.
        msgs = [str(e.get('msg', '')).removeprefix('Value error, ') for e in exc.errors()]
        detail = '; '.join(msgs) if len(msgs) > 0 else 'invalid request'
        return JSONResponse(status_code=400, content={'detail': detail})

    @app.middleware('http')
    async def dashboard_gate(request: Request, call_next):  # type: ignore[no-untyped-def]
        # The dashboard feature flag controls only the SPA bundle and SPA-only API routes;
        # CLI routes are always available. SPA fetch paths live under /api/dashboard/.
        path = request.url.path
        if not state.dashboard_enabled():
            if path == '/' or (not path.startswith('/api/') and _STATIC_DIR.exists()):
                return JSONResponse(status_code=404, content={'detail': 'dashboard disabled'})
            if path.startswith('/api/dashboard/') and path != '/api/dashboard/control':
                return JSONResponse(status_code=503, content={'detail': 'dashboard disabled'})
        return await call_next(request)

    if _STATIC_DIR.exists():
        # html=True makes StaticFiles serve index.html for any path that isn't an actual file,
        # which is what an SPA with client-side routing needs.
        app.mount('/', StaticFiles(directory=str(_STATIC_DIR), html=True), name='spa')

    return app

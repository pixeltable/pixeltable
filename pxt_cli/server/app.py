"""ThreadingHTTPServer host for the pxt daemon.

The server stays stdlib-only on top of pydantic (already a pixeltable dep) so the daemon
ships in the base wheel without an `extras` install. Each request runs in its own thread:
pixeltable calls are sync and release the GIL during SQL, so the threaded model gives
true concurrency without an asyncio stack.

Public entry point is `serve(port)`. Routes live in `routes.py` and register on the module
level `router` singleton.
"""

from __future__ import annotations

import http
import json
import logging
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pydantic

from pixeltable import exceptions as excs

from . import state
from .router import RawResponse, Request
from .routes import router

_logger = logging.getLogger('pixeltable.pxt_cli')

# Origins for which the daemon answers CORS preflight: the SPA's Vite dev server (5173)
# and the 8080 origin used by some sample deployments.
_DEV_ORIGINS = ('http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080')

# SPA bundle directory: the Vite build drops here.
_STATIC_DIR = Path(__file__).parent / 'static'
# Resolve presence once at import time: the bundle does not appear or vanish during the
# daemon's lifetime, so re-stat'ing per request would add a syscall to the hot path.
_HAS_STATIC_BUNDLE = _STATIC_DIR.exists()


class _DaemonHandler(BaseHTTPRequestHandler):
    """Dispatcher for /api/* JSON routes and static SPA files."""

    # Silence per-request access lines that BaseHTTPRequestHandler logs by default.
    def log_message(self, fmt: str, *args: Any) -> None:
        _logger.debug(fmt, *args)

    def handle(self) -> None:
        # Clients dropping the connection mid-response is normal (eg curl | head). Swallow
        # the resulting BrokenPipe so it doesn't reach the threading scaffolding's
        # default sys.excepthook and pollute the daemon log.
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(http.HTTPStatus.NO_CONTENT)
        self._write_cors_headers()
        self.end_headers()

    def do_GET(self) -> None:
        self._dispatch('GET')

    def do_POST(self) -> None:
        self._dispatch('POST')

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        url_path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        if not state.dashboard_enabled():
            # The dashboard feature flag controls only the SPA bundle and SPA-only API routes;
            # CLI routes are always available. SPA fetch paths live under /api/dashboard/.
            if url_path == '/' or (not url_path.startswith('/api/') and _HAS_STATIC_BUNDLE):
                self._send_json({'detail': 'dashboard disabled'}, http.HTTPStatus.NOT_FOUND)
                return
            if url_path.startswith('/api/dashboard/') and url_path != '/api/dashboard/control':
                self._send_json({'detail': 'dashboard disabled'}, http.HTTPStatus.SERVICE_UNAVAILABLE)
                return

        match = router.match(method, url_path)
        if match is None:
            # Static fallback: the SPA expects unknown non-/api/ paths to resolve to index.html
            # so client-side routing works. Reject if the SPA isn't bundled.
            if method == 'GET' and not url_path.startswith('/api/') and _HAS_STATIC_BUNDLE:
                self._serve_static(url_path)
                return
            self._send_json({'detail': 'not found'}, http.HTTPStatus.NOT_FOUND)
            return

        handler, path_params = match
        body_bytes = self._read_body() if method == 'POST' else b''
        req = Request(path_params=path_params, query=query, body_bytes=body_bytes)
        try:
            result = handler(req)
        except excs.Error as e:
            self._send_json({'detail': str(e), 'error_code': e.error_code.name}, e.http_status)
            return
        except pydantic.ValidationError as e:
            msgs = [str(err.get('msg', '')).removeprefix('Value error, ') for err in e.errors()]
            detail = '; '.join(m for m in msgs if m != '') or 'invalid request'
            self._send_json({'detail': detail}, http.HTTPStatus.BAD_REQUEST)
            return
        except Exception:
            _logger.exception('Unhandled error in %s %s', method, url_path)
            self._send_json({'detail': 'internal server error'}, http.HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if isinstance(result, RawResponse):
            self._send_raw(result)
        else:
            self._send_json(_to_jsonable(result))

    def _read_body(self) -> bytes:
        length_header = self.headers.get('Content-Length')
        if length_header is None:
            return b''
        try:
            length = int(length_header)
        except ValueError:
            return b''
        return self.rfile.read(length) if length > 0 else b''

    def _send_json(self, data: Any, status: int = http.HTTPStatus.OK) -> None:
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self._write_cors_headers()
        self.end_headers()
        self._safe_write(body)

    def _send_raw(self, resp: RawResponse) -> None:
        self.send_response(http.HTTPStatus(resp.status))
        self.send_header('Content-Type', resp.content_type)
        self.send_header('Content-Length', str(len(resp.body)))
        for k, v in resp.extra_headers.items():
            self.send_header(k, v)
        self._write_cors_headers()
        self.end_headers()
        self._safe_write(resp.body)

    def _write_cors_headers(self) -> None:
        origin = self.headers.get('Origin', '')
        if origin in _DEV_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Vary', 'Origin')

    def _serve_static(self, url_path: str) -> None:
        if url_path != '/':
            file_path = (_STATIC_DIR / url_path.lstrip('/')).resolve()
            # Containment check defends against `..`/symlink escapes from the static root.
            if _STATIC_DIR.resolve() in file_path.parents and file_path.is_file():
                self._send_file(file_path)
                return
        # html=True semantics: any path that isn't an actual file resolves to index.html so
        # SPA client-side routing works.
        index = _STATIC_DIR / 'index.html'
        if index.is_file():
            self._send_file(index)
            return
        self._send_json({'detail': 'not found'}, http.HTTPStatus.NOT_FOUND)

    def _send_file(self, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(http.HTTPStatus.OK)
        self.send_header('Content-Type', content_type or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        # Vite emits hashed asset filenames, so /assets/* contents are immutable per build.
        if 'assets' in file_path.parts:
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        self.end_headers()
        self._safe_write(body)

    def _safe_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


def _to_jsonable(result: Any) -> Any:
    if isinstance(result, pydantic.BaseModel):
        return result.model_dump(mode='json')
    return result


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that silences BrokenPipeError tracebacks at the connection level."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def serve(port: int) -> None:
    """Run the daemon on 127.0.0.1:port (blocks the calling thread)."""
    server = _QuietServer(('127.0.0.1', port), _DaemonHandler)
    _logger.info('pxt daemon listening on http://127.0.0.1:%s', port)
    try:
        server.serve_forever()
    finally:
        server.server_close()

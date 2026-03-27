"""
HTTP server for the Pixeltable Dashboard.

Uses stdlib `http.server.ThreadingHTTPServer`: every request runs in
its own thread, so synchronous bridge calls never block other requests.
No asyncio, no event loop, no third-party server dependency.

Designed to bind to 127.0.0.1 only: never expose to the network.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import sys
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, NamedTuple
from urllib.parse import parse_qs, unquote, urlparse

from pixeltable.dashboard import bridge

_logger = logging.getLogger('pixeltable.dashboard')

DASHBOARD_DIST_PATH = Path(__file__).parent / 'static'

_ALLOWED_ORIGINS = ('http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080')


class _RawResponse(NamedTuple):
    """Wrapper for non-JSON responses (e.g. CSV downloads)."""

    body: bytes
    content_type: str
    filename: str


RouteHandler = Callable[[str, dict], dict | list | _RawResponse]

_API_ROUTES: list[tuple[str, RouteHandler]] = []


def _api_route(prefix: str) -> Callable[[RouteHandler], RouteHandler]:
    """Decorator factory to register an API route."""

    def decorator(handler: RouteHandler) -> RouteHandler:
        _API_ROUTES.append((prefix, handler))
        return handler

    return decorator


@_api_route('/api/pixeltable-health')
def _(_path: str, _query: dict) -> dict:
    import pixeltable as pxt

    return {'status': 'ok', 'version': pxt.__version__}


@_api_route('/api/dirs')
def _(_path: str, _query: dict) -> list:
    return bridge.get_directory_tree()


@_api_route('/api/pipeline')
def _(_path: str, _query: dict) -> dict:
    return bridge.get_pipeline()


@_api_route('/api/status')
def _(_path: str, _query: dict) -> dict:
    return bridge.get_status()


@_api_route('/api/search')
def _(_path: str, query: dict) -> dict:
    q = query.get('q', '')
    if not q:
        return {'query': '', 'directories': [], 'tables': [], 'columns': []}
    limit = min(int(query.get('limit', '50')), 100)
    return bridge.search(q, limit=limit)


@_api_route('/api/tables/data')
def _(path: str, query: dict) -> dict:
    return bridge.get_table_data(
        path,
        offset=int(query.get('offset', '0')),
        limit=min(int(query.get('limit', '50')), 500),
        order_by=query.get('order_by'),
        order_desc=query.get('order_desc', 'false').lower() == 'true',
        errors_only=query.get('errors_only', 'false').lower() == 'true',
    )


@_api_route('/api/tables/meta')
def _(path: str, _query: dict) -> dict:
    return dict(bridge.get_table_metadata(path))


@_api_route('/api/tables/export')
def _(path: str, query: dict) -> _RawResponse:
    limit = min(int(query.get('limit', '100000')), 1_000_000)
    csv_bytes = bridge.export_table_csv(path, limit=limit)
    safe_name = path.replace('/', '_')
    return _RawResponse(csv_bytes, 'text/csv; charset=utf-8', f'{safe_name}.csv')


class _DashboardHandler(BaseHTTPRequestHandler):
    """Handles GET requests: API routes + SPA static files."""

    # Silence per-request log lines from BaseHTTPRequestHandler
    def log_message(self, fmt: str, *args: Any) -> None:
        _logger.debug(fmt, *args)

    def _cors_origin(self) -> str:
        origin = self.headers.get('Origin', '')
        return origin if origin in _ALLOWED_ORIGINS else ''

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header('Access-Control-Allow-Origin', self._cors_origin())
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/api/'):
            self.__handle_api(path, parsed.query)
            return

        self.__serve_static(path)

    def do_HEAD(self) -> None:
        self.do_GET()

    # Block all other methods
    def do_POST(self) -> None:
        self.__method_not_allowed()

    def do_PUT(self) -> None:
        self.__method_not_allowed()

    def do_DELETE(self) -> None:
        self.__method_not_allowed()

    def do_PATCH(self) -> None:
        self.__method_not_allowed()

    def __method_not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.__safe_write(b'Method not allowed')

    def __handle_api(self, path: str, raw_query: str) -> None:
        query = {k: v[0] for k, v in parse_qs(raw_query).items()} if raw_query else {}

        for prefix, handler in _API_ROUTES:
            if path.startswith(prefix):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        path = path.removeprefix(prefix)
                        path = path.removeprefix('/')  # Do this separately since trailing path is optional
                        result = handler(unquote(path), query)
                    if isinstance(result, _RawResponse):
                        self.__send_raw(result)
                    else:
                        self.__send_json(result)
                except Exception as e:
                    _logger.exception('Error handling %s', path)
                    safe = str(e).split('\n')[0][:200] if str(e) else f'{type(e).__name__}: (no message)'
                    self.__send_json({'error': safe}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

        self.__send_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)

    def __send_json(self, data: dict | list, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, default=str).encode()
        allowed = self._cors_origin()

        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'SAMEORIGIN')
        self.send_header('Referrer-Policy', 'strict-origin-when-cross-origin')
        if allowed:
            self.send_header('Access-Control-Allow-Origin', allowed)
        self.end_headers()
        self.__safe_write(body)

    def __send_raw(self, resp: _RawResponse) -> None:
        allowed = self._cors_origin()

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', resp.content_type)
        self.send_header('Content-Length', str(len(resp.body)))
        self.send_header('Content-Disposition', f'attachment; filename="{resp.filename}"')
        self.send_header('Cache-Control', 'no-store')
        if allowed:
            self.send_header('Access-Control-Allow-Origin', allowed)
        self.end_headers()
        self.__safe_write(resp.body)

    def __serve_static(self, path: str) -> None:
        # Try to serve the exact file (for /assets/*, /logo.png, etc.)
        if path != '/':
            file_path = DASHBOARD_DIST_PATH / path.lstrip('/')
            if file_path.exists() and file_path.is_file() and DASHBOARD_DIST_PATH in file_path.resolve().parents:
                self.__send_file(file_path)
                return

        # Everything else → index.html (SPA routing)
        index = DASHBOARD_DIST_PATH / 'index.html'
        if index.exists():
            self.__send_file(index)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, 'Dashboard not found')

    def __send_file(self, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        if '/assets/' in str(file_path):
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        self.end_headers()
        self.__safe_write(body)

    def __safe_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that silences BrokenPipeError tracebacks."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def run_server(port: int) -> None:
    """Run the dashboard server (blocks the calling thread).

    Uses `ThreadingHTTPServer` — each request is handled in its own
    thread, so synchronous Pixeltable calls never block other requests.

    Called from a daemon thread spawned by `_start_dashboard_background`
    in `globals.py`.

    Args:
        host: Address to bind to (default `127.0.0.1`; never use `0.0.0.0`).
        port: Port number.
    """
    assert DASHBOARD_DIST_PATH.exists(), f'Static site distribution not found at: {DASHBOARD_DIST_PATH}'

    server = _QuietServer(('127.0.0.1', port), _DashboardHandler)
    server.daemon_threads = True
    _logger.info('Pixeltable Dashboard initialized on http://127.0.0.1:%s', port)

    try:
        server.serve_forever()
    except Exception as exc:
        _logger.error('Dashboard server terminated unexpectedly: %s', exc)
        raise
    finally:
        server.server_close()

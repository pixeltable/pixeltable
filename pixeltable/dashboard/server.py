"""
HTTP server for the Pixeltable Dashboard.

Uses stdlib ``http.server.ThreadingHTTPServer`` — every request runs in
its own thread, so synchronous bridge calls never block other requests.
No asyncio, no event loop, no third-party server dependency.

Designed to bind to 127.0.0.1 only — never expose to the network.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import parse_qs, unquote, urlparse

from pixeltable.dashboard import bridge

_logger = logging.getLogger('pixeltable.dashboard')

DASHBOARD_DIST_PATH = Path(__file__).parent / 'static'

_ALLOWED_ORIGINS = frozenset(
    {'http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080'}
)

# ── Routing table ────────────────────────────────────────────────────────────
#
# Each entry: (compiled regex, handler function)
# Handler signature: (match: re.Match, query: dict[str, str]) -> (dict | list)
# Handlers are plain sync functions — they run in the request's own thread.

Route = tuple[re.Pattern[str], Any]  # (pattern, handler_fn)


class _RawResponse(NamedTuple):
    """Wrapper for non-JSON responses (e.g. CSV downloads)."""

    body: bytes
    content_type: str
    filename: str


def _route_health(_m: re.Match, _q: dict) -> dict:
    import pixeltable

    return {'status': 'ok', 'version': getattr(pixeltable, '__version__', 'unknown')}


def _route_dirs(_m: re.Match, _q: dict) -> list:
    return bridge.get_directory_tree()


def _route_pipeline(_m: re.Match, _q: dict) -> dict:
    return bridge.get_pipeline()


def _route_status(_m: re.Match, _q: dict) -> dict:
    return bridge.get_status()


def _route_search(_m: re.Match, q: dict) -> dict:
    query = q.get('q', '')
    if not query:
        return {'query': '', 'directories': [], 'tables': [], 'columns': []}
    limit = min(int(q.get('limit', '50')), 100)
    return bridge.search(query, limit=limit)


def _route_table_data(m: re.Match, q: dict) -> dict:
    path = unquote(m.group('path'))
    return bridge.get_table_data(
        path,
        offset=int(q.get('offset', '0')),
        limit=min(int(q.get('limit', '50')), 500),
        order_by=q.get('order_by'),
        order_desc=q.get('order_desc', 'false').lower() == 'true',
        errors_only=q.get('errors_only', 'false').lower() == 'true',
    )


def _route_table_meta(m: re.Match, _q: dict) -> dict:
    return bridge.get_table_metadata(unquote(m.group('path')))


def _route_table_export(m: re.Match, q: dict) -> _RawResponse:
    path = unquote(m.group('path'))
    limit = min(int(q.get('limit', '100000')), 1_000_000)
    csv_bytes = bridge.export_table_csv(path, limit=limit)
    safe_name = path.replace('/', '_')
    return _RawResponse(csv_bytes, 'text/csv; charset=utf-8', f'{safe_name}.csv')


# Order matters: more specific patterns first.
_API_ROUTES: list[Route] = [
    (re.compile(r'^/api/health$'), _route_health),
    (re.compile(r'^/api/dirs$'), _route_dirs),
    (re.compile(r'^/api/pipeline$'), _route_pipeline),
    (re.compile(r'^/api/status$'), _route_status),
    (re.compile(r'^/api/search$'), _route_search),
    (re.compile(r'^/api/tables/(?P<path>.+)/export$'), _route_table_export),
    (re.compile(r'^/api/tables/(?P<path>.+)/data$'), _route_table_data),
    (re.compile(r'^/api/tables/(?P<path>.+)$'), _route_table_meta),
]


# ── Request Handler ──────────────────────────────────────────────────────────


class _DashboardHandler(BaseHTTPRequestHandler):
    """Handles GET requests: API routes + SPA static files."""

    # Silence per-request log lines from BaseHTTPRequestHandler
    def log_message(self, fmt: str, *args: Any) -> None:
        _logger.debug(fmt, *args)

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_OPTIONS(self) -> None:
        origin = self.headers.get('Origin', '')
        allowed = origin if origin in _ALLOWED_ORIGINS else ''
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header('Access-Control-Allow-Origin', allowed)
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        # ── API routes ────────────────────────────────────────────────
        if path.startswith('/api/'):
            self._handle_api(path, parsed.query)
            return

        # ── Static files ──────────────────────────────────────────────
        self._serve_static(path)

    def do_HEAD(self) -> None:
        self.do_GET()

    # Block all other methods
    def do_POST(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def do_PATCH(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self._safe_write(b'Method not allowed')

    # ── API handling ──────────────────────────────────────────────────

    def _handle_api(self, path: str, raw_query: str) -> None:
        query = {k: v[0] for k, v in parse_qs(raw_query).items()} if raw_query else {}

        for pattern, handler in _API_ROUTES:
            match = pattern.match(path)
            if match:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        result = handler(match, query)
                    if isinstance(result, _RawResponse):
                        self._send_raw(result)
                    else:
                        self._send_json(result)
                except Exception as e:
                    _logger.exception('Error handling %s', path)
                    safe = str(e).split('\n')[0][:200] if str(e) else f'{type(e).__name__}: (no message)'
                    self._send_json({'error': safe}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

        self._send_json({'error': 'Not found'}, status=HTTPStatus.NOT_FOUND)

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, default=str).encode()
        origin = self.headers.get('Origin', '')
        allowed = origin if origin in _ALLOWED_ORIGINS else ''

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
        self._safe_write(body)

    def _send_raw(self, resp: _RawResponse) -> None:
        origin = self.headers.get('Origin', '')
        allowed = origin if origin in _ALLOWED_ORIGINS else ''

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', resp.content_type)
        self.send_header('Content-Length', str(len(resp.body)))
        self.send_header('Content-Disposition', f'attachment; filename="{resp.filename}"')
        self.send_header('Cache-Control', 'no-store')
        if allowed:
            self.send_header('Access-Control-Allow-Origin', allowed)
        self.end_headers()
        self._safe_write(resp.body)

    # ── Static file serving ───────────────────────────────────────────

    def _serve_static(self, path: str) -> None:
        if not DASHBOARD_DIST_PATH.exists():
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(
                b'<h1>Pixeltable Dashboard</h1>'
                b'<p>Frontend not built. Run: <code>cd dashboard && npm install && npm run build</code></p>'
                b'<p><a href="/api/dirs">/api/dirs</a></p>'
            )
            return

        # Try to serve the exact file (for /assets/*, /logo.png, etc.)
        if path != '/':
            file_path = DASHBOARD_DIST_PATH / path.lstrip('/')
            if file_path.exists() and file_path.is_file() and DASHBOARD_DIST_PATH in file_path.resolve().parents:
                self._send_file(file_path)
                return

        # Everything else → index.html (SPA routing)
        index = DASHBOARD_DIST_PATH / 'index.html'
        if index.exists():
            self._send_file(index)
        else:
            self.send_error(HTTPStatus.NOT_FOUND, 'Dashboard not found')

    def _send_file(self, file_path: Path) -> None:
        content_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Type', content_type or 'application/octet-stream')
        self.send_header('Content-Length', str(len(body)))
        if '/assets/' in str(file_path):
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
        self.end_headers()
        self._safe_write(body)

    def _safe_write(self, data: bytes) -> None:
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ── Server entry point ───────────────────────────────────────────────────────


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that silences BrokenPipeError tracebacks."""

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def run_server(host: str = '127.0.0.1', port: int = 8080) -> None:
    """Run the dashboard server (blocks the calling thread).

    Uses ``ThreadingHTTPServer`` — each request is handled in its own
    thread, so synchronous Pixeltable calls never block other requests.

    Called from a daemon thread spawned by ``_start_dashboard_background``
    in ``globals.py``.

    Args:
        host: Address to bind to (default ``127.0.0.1``; never use ``0.0.0.0``).
        port: Port number.
    """
    server = _QuietServer((host, port), _DashboardHandler)
    server.daemon_threads = True
    _logger.info('Pixeltable Dashboard serving on http://%s:%s', host, port)

    try:
        server.serve_forever()
    finally:
        server.server_close()

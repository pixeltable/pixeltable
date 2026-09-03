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
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pydantic

from pixeltable import exceptions as excs
from pixeltable.config import Config, env_var_name

from .daemon_state import compare_env_values, config_fingerprint, state as daemon_state
from .router import Method, RawResponse, Request
from .routes import router

_logger = logging.getLogger('pixeltable.pixeltable_cli')

# Origins for which the daemon answers CORS preflight: the SPA's Vite dev server (5173)
# and the 8080 origin used by some sample deployments.
_DEV_ORIGINS = ('http://localhost:5173', 'http://127.0.0.1:5173', 'http://localhost:8080', 'http://127.0.0.1:8080')

# SPA bundle directory: the Vite build drops here.
_STATIC_DIR = Path(__file__).parent / 'static'
# Resolve presence once at import time: the bundle does not appear or vanish during the
# daemon's lifetime, so re-stat'ing per request would add a syscall to the hot path.
_HAS_STATIC_BUNDLE = _STATIC_DIR.exists()


# Header carrying the caller's env fingerprint: {env var name: hash of its value}, no values.
_ENV_HEADER = 'x-pxt-env-fingerprint'


def _changed_settings(names: list[str]) -> str:
    """One line per changed setting: how the file or the environment holding it spells it, and where it is."""
    config = Config.get()
    keys = {env_var_name(ck.section, ck.key): ck for ck in config.env_keys()}
    lines: list[str] = []
    for name in names:
        ck = keys.get(name)
        lines.append(
            f'  {config.describe_setting(ck.section, ck.key)}' if ck is not None else f'  {name}, no longer set'
        )
    return '\n'.join(lines)


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

    def _dispatch(self, method: Method) -> None:
        # Users edit the config file directly, so pick up an edit here rather than at the next daemon
        # restart. Doing it once per request means a request sees one consistent set of values.
        parsed = urlparse(self.path)
        url_path = unquote(parsed.path)
        query = parse_qs(parsed.query, keep_blank_values=True)

        # /api/health does not need the config file
        if url_path != '/api/health':
            try:
                if Config.reload_if_changed():
                    _logger.info('Reloaded %s', Config.get().config_file)
            except excs.Error as e:
                self._send_json({'detail': str(e), 'error_code': e.error_code.name}, e.http_status)
                return
            except Exception as e:
                self._send_json(
                    {'detail': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()},
                    http.HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return

        handler = router.match(method, url_path)
        if handler is None:
            # Static fallback: the SPA expects unknown non-/api/ paths to resolve to index.html
            # so client-side routing works. Reject if the SPA isn't bundled.
            if method == 'GET' and not url_path.startswith('/api/') and _HAS_STATIC_BUNDLE:
                self._serve_static(url_path)
                return
            self._send_json({'detail': 'not found'}, http.HTTPStatus.NOT_FOUND)
            return

        if method == 'POST':
            body_bytes = self._read_body()
            if body_bytes is None:
                # 413 already sent by _read_body
                return
        else:
            body_bytes = b''
        req = Request(query=query, body_bytes=body_bytes, headers={k.lower(): v for k, v in self.headers.items()})
        if router.checks_env(method, url_path) and not self._env_values_agree(req):
            return

        # health and status requests finish instantly
        request_id = (
            None if url_path in ('/api/health', '/api/status') else daemon_state.begin_request(method, url_path)
        )
        try:
            result = handler(req)
        except excs.Error as e:
            self._send_json({'detail': str(e), 'error_code': e.error_code.name}, e.http_status)
            return
        except Exception as e:
            # log the full request target (self.path includes the query string) plus the resolved catalog
            # paths, so the affected path is visible whether it arrived in the query string or the body
            _logger.exception('Unhandled error in %s %s (paths=%s)', method, self.path, req.resolved_paths)
            # Return the exception message and daemon-side traceback rather than a bare 'internal server
            # error': the daemon serves only localhost for the same user, so this leaks nothing, and it
            # gives the user a concrete failure to act on and report.
            self._send_json(
                {'detail': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc()},
                http.HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        finally:
            if request_id is not None:
                daemon_state.end_request(request_id)

        if isinstance(result, RawResponse):
            self._send_raw(result)
        else:
            self._send_json(_to_jsonable(result))

    def _env_values_agree(self, req: Request) -> bool:
        """Whether this daemon's config values are the ones it recorded and the caller expects."""
        current = config_fingerprint()
        changed = daemon_state.changed_env_vars(current)
        if len(changed) > 0:
            # api clients are built from these values and cached per worker thread, so serving with the new
            # ones takes a new process
            self._send_json(
                {
                    'detail': f'configuration has changed since the daemon started:\n{_changed_settings(changed)}\n\n'
                    'The daemon reads configuration once at startup. '
                    'Run `pxt daemon restart` to pick up the new configuration.',
                    'error_code': 'STALE_CONFIG',
                },
                http.HTTPStatus.CONFLICT,
            )
            return False
        return self._caller_env_values_agree(req, current)

    def _caller_env_values_agree(self, req: Request, current: dict[str, str]) -> bool:
        """Whether the caller's config values match this daemon's, sending a 409 naming the difference if not."""
        header = req.headers.get(_ENV_HEADER)
        if header is None:
            return True
        try:
            caller: dict[str, str] = json.loads(header)
        except json.JSONDecodeError:
            return True  # an unparseable fingerprint is no evidence of disagreement
        missing, differing = compare_env_values(caller, current)
        if len(missing) == 0 and len(differing) == 0:
            return True

        detail: list[str] = []
        if len(missing) > 0:
            detail.append(f'  set here but not in the daemon: {", ".join(missing)}')
        if len(differing) > 0:
            detail.append(f'  set to a different value in the daemon: {", ".join(differing)}')
        self._send_json(
            {
                'detail': 'the daemon started with a different environment:\n'
                + '\n'.join(detail)
                + '\n\nThe daemon reads the environment once at startup. '
                'Run `pxt daemon restart` to pick up your environment.',
                'error_code': 'STALE_CONFIG',
            },
            http.HTTPStatus.CONFLICT,
        )
        return False

    # Cap request body size. Localhost-only daemon, but a misbehaving local client could
    # otherwise pin a thread on a multi-GB read or exhaust memory in pydantic validation.
    _MAX_BODY_BYTES = 1 * 1024 * 1024

    def _read_body(self) -> bytes | None:
        """Return the request body, or None if the caller exceeded the size cap (in which
        case a 413 has already been sent)."""
        length_header = self.headers.get('Content-Length')
        if length_header is None:
            return b''
        try:
            length = int(length_header)
        except ValueError:
            return b''
        if length > self._MAX_BODY_BYTES:
            self._send_json(
                {'detail': f'request body exceeds the {self._MAX_BODY_BYTES}-byte limit'},
                http.HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return None
        return self.rfile.read(length) if length > 0 else b''

    def _write_hardening_headers(self) -> None:
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')

    def _send_json(self, data: Any, status: int = http.HTTPStatus.OK) -> None:
        body = json.dumps(data, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self._write_hardening_headers()
        self._write_cors_headers()
        self.end_headers()
        self._safe_write(body)

    def _send_raw(self, resp: RawResponse) -> None:
        self.send_response(http.HTTPStatus(resp.status))
        self.send_header('Content-Type', resp.content_type)
        self.send_header('Content-Length', str(len(resp.body)))
        self.send_header('Cache-Control', 'no-store')
        self._write_hardening_headers()
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
            # Containment check defends against path traversal and symlink escapes from the static root.
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
    if isinstance(result, list):
        return [_to_jsonable(item) for item in result]
    return result


class _QuietServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that silences BrokenPipeError tracebacks at the connection level."""

    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


def bind(port: int, host: str = '127.0.0.1') -> _QuietServer:
    """Bind the listen socket. Raises OSError if the port is already taken."""
    return _QuietServer((host, port), _DaemonHandler)


def run(server: _QuietServer) -> None:
    """Serve forever on a server bound by bind() (blocks the calling thread)."""
    # before the first request: a request that reloads the config file would otherwise be the one to record
    # the baseline, and would record the file's new values as the ones this process serves with
    daemon_state.record_env_fingerprint()
    _logger.info('pxt daemon listening on http://%s:%s', *server.server_address)
    try:
        server.serve_forever()
    finally:
        server.server_close()

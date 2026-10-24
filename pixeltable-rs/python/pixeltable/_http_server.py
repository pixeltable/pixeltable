"""The local HTTP file server behind media links in HTML renderings of results (as upstream's runtime starts one).

The server root is the file system root: `/home/media/foo.mp4` is served at `http://127.0.0.1:{port}/home/media/foo.mp4`.
It listens on 127.0.0.1 only, on a port the OS picks, and starts on first use.
"""

from __future__ import annotations

import http.server
import logging
import pathlib
import threading
import urllib.request
from typing import Any

_logger = logging.getLogger('pixeltable')

_lock = threading.Lock()
_address: str | None = None


class _AbsolutePathHandler(http.server.SimpleHTTPRequestHandler):
    """Serves absolute paths, not just the current directory."""

    def translate_path(self, path: str) -> str:
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        return str(pathlib.Path(urllib.request.url2pathname(path)))

    def log_message(self, format: str, *args: Any) -> None:
        _logger.info(format % args)


class _QuietHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        import traceback

        _logger.error(f'Exception while serving {client_address}:\n{traceback.format_exc()}')


def http_address() -> str:
    """The root address of the file server, starting it if needed."""
    global _address
    with _lock:
        if _address is None:
            httpd = _QuietHTTPServer(('127.0.0.1', 0), _AbsolutePathHandler)
            _address = f'http://127.0.0.1:{httpd.server_address[1]}'
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return _address


def file_uri(file_path: str) -> str:
    """The URL of a local file on the file server."""
    abs_path = pathlib.Path(file_path)
    assert abs_path.is_absolute()
    return f'{http_address()}{urllib.request.pathname2url(str(abs_path))}'

import http
import http.server
import logging
import pathlib
import urllib
from typing import Any

_logger = logging.getLogger('pixeltable.http.server')


def get_file_uri(http_address: str, file_path: str) -> str:
    """Get the URI for a file path, with the given prefix.
    Used in the client to generate a URI
    """
    abs_path = pathlib.Path(file_path)
    assert abs_path.is_absolute()
    url = urllib.request.pathname2url(str(abs_path))
    return f'{http_address}{url}'


class AbsolutePathHandler(http.server.SimpleHTTPRequestHandler):
    """Serves all absolute paths, not just the current directory"""

    def translate_path(self, path: str) -> str:
        """
        Translate a /-separated PATH to the local filename syntax.
        overrides http.server.SimpleHTTPRequestHandler.translate_path

        This is only useful for file serving.

        Code initially taken from there:
        https://github.com/python/cpython/blob/f5406ef454662b98df107775d18ff71ae6849618/Lib/http/server.py#L834
        """
        _logger.info(f'translate path {path=}')
        # abandon query parameters, taken from http.server.SimpleHTTPRequestHandler
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]

        path = pathlib.Path(urllib.request.url2pathname(path))
        return str(path)

    def log_message(self, format: str, *args: Any) -> None:
        """override logging to stderr in http.server.BaseHTTPRequestHandler"""
        message = format % args
        _logger.info(message.translate(self._control_char_table))  # type: ignore[attr-defined]


class LoggingHTTPServer(http.server.ThreadingHTTPServer):
    """Avoids polluting stdout and stderr"""

    def handle_error(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        """override socketserver.TCPServer.handle_error which prints directly to sys.stderr"""
        import traceback

        _logger.error(
            f'Exception occurred during processing of {request=} from {client_address=}\
            \nbacktrace:\n{traceback.format_exc()}\n----\n'
        )


def make_server(address: str, port: int) -> http.server.HTTPServer:
    """Create a file server with pixeltable specific config"""
    return LoggingHTTPServer((address, port), AbsolutePathHandler)


if __name__ == '__main__':
    httpd = make_server('127.0.0.1', 8000)
    print(f'about to server HTTP on {httpd.server_address}')
    httpd.serve_forever()

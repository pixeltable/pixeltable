import http.server
import io
import logging
import os
import pathlib
import secrets
import urllib.request
from typing import Any, BinaryIO

_logger = logging.getLogger(__name__)

# content types that a browser renders as a document able to run scripts, along with every type ending in +xml
_ACTIVE_CONTENT_TYPES = ('text/html', 'application/xml', 'text/xml')


def get_file_uri(http_address: str, file_path: str) -> str:
    """Get the URI for a file path, with the given prefix.
    Used in the client to generate a URI
    """
    abs_path = pathlib.Path(file_path)
    assert abs_path.is_absolute()
    url = urllib.request.pathname2url(str(abs_path))
    return f'{http_address}{url}'


class AbsolutePathHandler(http.server.SimpleHTTPRequestHandler):
    """Serves all absolute paths, not just the current directory, to requests that carry the server's token"""

    server: 'LoggingHTTPServer'

    def send_head(self) -> io.BytesIO | BinaryIO | None:
        """
        Refuses a request without the token, before any filesystem access.

        Unlike the daemon, the server accepts any Host, because the proxy of a port forwarder such as Codespaces
        sends its own host name. Only the token stops a web page that rebinds its host name to 127.0.0.1, so a
        URL leaked from shared notebook output works from any page for the life of this process.
        """
        prefix = f'/{self.server.token}/'
        if not secrets.compare_digest(self.path[: len(prefix)].encode(), prefix.encode()):
            self.send_error(http.HTTPStatus.FORBIDDEN)
            return None
        return super().send_head()

    def translate_path(self, path: str) -> str:
        """
        Translate a /-separated PATH to the local filename syntax.
        overrides http.server.SimpleHTTPRequestHandler.translate_path

        This is only useful for file serving.

        Code initially taken from there:
        https://github.com/python/cpython/blob/f5406ef454662b98df107775d18ff71ae6849618/Lib/http/server.py#L834
        """
        # only send_head() checks the token, and it calls this only for a path that starts with it
        path = path.removeprefix(f'/{self.server.token}')
        _logger.info(f'translate path {path=}')
        # abandon query parameters, taken from http.server.SimpleHTTPRequestHandler
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]

        return str(pathlib.Path(urllib.request.url2pathname(path)))

    def list_directory(self, path: str | os.PathLike[str]) -> io.BytesIO | None:
        """Refuses to list a directory"""
        self.send_error(http.HTTPStatus.NOT_FOUND)
        return None

    def send_header(self, keyword: str, value: str) -> None:
        super().send_header(keyword, value)
        # a script in a served document would run in this server's origin, with the token in its own URL
        ctype = value.partition(';')[0]
        if keyword.lower() == 'content-type' and (ctype in _ACTIVE_CONTENT_TYPES or ctype.endswith('+xml')):
            super().send_header('Content-Security-Policy', 'sandbox')

    def log_message(self, format: str, *args: Any) -> None:
        """override logging to stderr in http.server.BaseHTTPRequestHandler, and keep the token out of the log"""
        message = (format % args).replace(self.server.token, '<token>')
        _logger.info(message.translate(self._control_char_table))  # type: ignore[attr-defined]


class LoggingHTTPServer(http.server.ThreadingHTTPServer):
    """Avoids polluting stdout and stderr"""

    # the first segment of every served path
    token: str

    @property
    def http_address(self) -> str:
        """The address that get_file_uri() prefixes to a file's absolute path"""
        host, port = self.server_address[:2]
        assert isinstance(host, str)
        return f'http://{host}:{port}/{self.token}'

    def handle_error(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        """override socketserver.TCPServer.handle_error which prints directly to sys.stderr"""
        import traceback

        _logger.error(
            f'Exception occurred during processing of {request=} from {client_address=}\
            \nbacktrace:\n{traceback.format_exc()}\n----\n'
        )


def make_server(address: str, port: int) -> LoggingHTTPServer:
    """Create a file server with pixeltable specific config"""
    server = LoggingHTTPServer((address, port), AbsolutePathHandler)
    server.token = secrets.token_urlsafe(32)
    return server


if __name__ == '__main__':
    httpd = make_server('127.0.0.1', 8000)
    print(f'about to server HTTP on {httpd.http_address}')
    httpd.serve_forever()

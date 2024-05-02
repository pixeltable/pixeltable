import http
import http.server
import logging
import urllib
import posixpath
import pathlib
import re
import os

_logger = logging.getLogger('pixeltable.http.server')

_regex = re.compile(r'^([A-Za-z]:)?(/.*)?$')

def get_file_uri(http_address: str, file_path: str) -> str:
    """Get the URI for a file path, with the given prefix.
        Used in the client to generate a URI
    """
    abs_path = pathlib.Path(file_path)
    assert abs_path.is_absolute()
    # for windows, replace '\\' with '/', keep the drive letter
    path_normalized = str(abs_path).replace(os.sep, '/')
    quoted = urllib.parse.quote(path_normalized, safe=':/')
    return f'{http_address}/{quoted}'


class AbsolutePathHandler(http.server.SimpleHTTPRequestHandler):
    """Serves all absolute paths, not just the current directory"""

    def __init__(self, *args, **kwargs):
        self.default_root = pathlib.Path(pathlib.Path('.').absolute().anchor)
        # in windows will be something like 'C:/', in posix '/', but dont want to assume C
        super().__init__(*args, directory=self.default_root, **kwargs)

    def translate_path(self, path):
        """
            Translate a /-separated PATH to the local filename syntax.
            overrides http.server.SimpleHTTPRequestHandler.translate_path
            it will translate http://localhost/c:/foo/bar to c:/foo/bar

            This is only useful for file serving.

            Code initially taken from there:
            https://github.com/python/cpython/blob/f5406ef454662b98df107775d18ff71ae6849618/Lib/http/server.py#L834
        """
        # abandon query parameters
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]

        try:
            path = urllib.parse.unquote(path, errors='surrogatepass')
        except UnicodeDecodeError:
            path = urllib.parse.unquote(path)

        path = posixpath.normpath(path).lstrip('/')  # will remove double slashes
        matches = _regex.match(path)
        if not matches:
            raise Exception(f'non-conforming path {path=}')

        (volume, remainder) = list(matches.groups())
        volume = self.default_root if not volume else pathlib.Path(volume + '/')
        remainder = remainder if remainder is not None else ''
        path = volume / remainder
        # print(f'{path=}, {path.exists()=} {path.is_dir()=} {path.is_absolute()=}')
        if not path.is_absolute():
            raise Exception(f'need absolute path. got {path=}')

        return str(path)

    def log_message(self, format, *args) -> None:
        """override logging to stderr in http.server.BaseHTTPRequestHandler"""
        message = format % args
        _logger.info(message.translate(self._control_char_table))


class LoggingHTTPServer(http.server.ThreadingHTTPServer):
    """Avoids polluting stdout and stderr"""

    def handle_error(self, request, client_address) -> None:
        """override socketserver.TCPServer.handle_error which prints directly to sys.stderr"""
        import traceback

        _logger.error(
            f'Exception occurred during processing of {request=} from {client_address=}\
            \nbacktrace:\n{traceback.format_exc()}\n----\n'
        )


def make_server(address: str, port: int) -> http.server.HTTPServer:
    """Create a file server with pixeltable specific config """
    return LoggingHTTPServer((address, port), AbsolutePathHandler)


if __name__ == '__main__':
    httpd = make_server('127.0.0.1', 8000)
    print(f'about to server HTTP on {httpd.server_address}')
    httpd.serve_forever()
import http
import logging

_logger = logging.getLogger('pixeltable.http.server')


class FixedRootHandler(http.server.SimpleHTTPRequestHandler):
    """Serves everything wrt. system root"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='/', **kwargs)

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

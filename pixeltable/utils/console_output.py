import contextlib
import contextvars
import logging
import threading
from typing import Iterator, TextIO

from pixeltable import exceptions as excs


def map_level(verbosity: int) -> int:
    """
        Map verbosity level to logging level.
        0 - minimum logging - warn and above
        1 - default logging - info and above
        2 - more logging - debug and above
    Args:

    Returns:
        Logging level as integer
    """
    if verbosity == 0:
        return logging.WARN
    if verbosity == 1:
        return logging.INFO
    if verbosity == 2:
        return logging.DEBUG

    raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid verbosity level: {verbosity}')


class ConsoleOutputHandler(logging.StreamHandler):
    def __init__(self, stream: TextIO):
        super().__init__(stream)

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg.endswith('\n'):
            self.stream.write(record.msg)
        else:
            self.stream.write(record.msg + '\n')


class ConsoleMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, 'user_visible', False)


class ConsoleLogger(logging.LoggerAdapter):
    def __init__(self, logger: logging.Logger):
        super().__init__(logger, extra={'user_visible': True})


_captured: contextvars.ContextVar[list[tuple[int, str]] | None] = contextvars.ContextVar(
    'captured_console', default=None
)


class _CaptureHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        lines = _captured.get()
        if lines is not None and getattr(record, 'user_visible', False):
            lines.append((record.levelno, record.getMessage()))


_capture_handler = _CaptureHandler()
_capture_handler_lock = threading.Lock()


@contextlib.contextmanager
def capture_console() -> Iterator[list[tuple[int, str]]]:
    """Collect the (level, message) of each console message logged in the current context.

    Messages still go to the regular handlers. A thread does not inherit the context, so messages logged on a
    worker thread are not collected.
    """
    logger = logging.getLogger('pixeltable')
    with _capture_handler_lock:
        if _capture_handler not in logger.handlers:
            logger.addHandler(_capture_handler)
    lines: list[tuple[int, str]] = []
    token = _captured.set(lines)
    try:
        yield lines
    finally:
        _captured.reset(token)

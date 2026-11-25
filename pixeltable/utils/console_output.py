import contextlib
import contextvars
import logging
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


class ConsoleCollector:
    """Console messages collected in memory, up to max_chars characters of message text.

    Once a message would exceed the cap, it and all later ones are dropped, and messages() ends with a warning
    that says how many.
    """

    max_chars: int
    _messages: list[tuple[int, str]]
    _num_chars: int
    _num_dropped: int

    def __init__(self, max_chars: int = 1_000_000):
        self.max_chars = max_chars
        self._messages = []
        self._num_chars = 0
        self._num_dropped = 0

    def add(self, level: int, msg: str) -> None:
        if self._num_dropped == 0 and self._num_chars + len(msg) <= self.max_chars:
            self._messages.append((level, msg))
            self._num_chars += len(msg)
        else:
            self._num_dropped += 1

    def messages(self) -> list[tuple[int, str]]:
        """The collected (level, message) pairs, in the order they were added."""
        if self._num_dropped == 0:
            return list(self._messages)
        note = f'{self._num_dropped} more console message{"" if self._num_dropped == 1 else "s"} omitted'
        return [*self._messages, (logging.WARNING, note)]


_collector: contextvars.ContextVar[ConsoleCollector | None] = contextvars.ContextVar('console_collector', default=None)


def current_collector() -> ConsoleCollector | None:
    return _collector.get()


@contextlib.contextmanager
def collect_console(collector: ConsoleCollector) -> Iterator[ConsoleCollector]:
    """Make collector receive the console messages logged in the current context.

    A thread does not inherit the context, so messages logged on a worker thread are not collected.
    """
    token = _collector.set(collector)
    try:
        yield collector
    finally:
        _collector.reset(token)

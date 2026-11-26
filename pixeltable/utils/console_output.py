import collections
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
    """Console messages collected in memory, up to about max_chars characters of message text.

    Past the cap, messages are dropped from the middle: the collector keeps the first and the last max_chars / 2
    characters' worth, and messages() inserts a warning between them that says how many were omitted. A message
    longer than max_chars / 2 is truncated.
    """

    max_chars: int
    _head: list[tuple[str, int]]
    _head_chars: int
    _tail: collections.deque[tuple[str, int]]
    _tail_chars: int
    _num_dropped: int

    def __init__(self, max_chars: int = 1_000_000):
        self.max_chars = max_chars
        self._head = []
        self._head_chars = 0
        self._tail = collections.deque()
        self._tail_chars = 0
        self._num_dropped = 0

    def add(self, msg: str, level: int) -> None:
        half = self.max_chars // 2
        if len(msg) > half:
            msg = msg[:half] + ' ... (truncated)'
        if len(self._tail) == 0 and (len(self._head) == 0 or self._head_chars + len(msg) <= half):
            self._head.append((msg, level))
            self._head_chars += len(msg)
            return
        self._tail.append((msg, level))
        self._tail_chars += len(msg)
        while self._tail_chars > half and len(self._tail) > 1:
            dropped, _ = self._tail.popleft()
            self._tail_chars -= len(dropped)
            self._num_dropped += 1

    def messages(self) -> list[tuple[str, int]]:
        """The collected (message, level) pairs, in the order they were added."""
        if self._num_dropped == 0:
            return [*self._head, *self._tail]
        note = f'... {self._num_dropped} console message{"" if self._num_dropped == 1 else "s"} omitted ...'
        return [*self._head, (note, logging.WARNING), *self._tail]


_collector: contextvars.ContextVar[ConsoleCollector | None] = contextvars.ContextVar('console_collector', default=None)


def get_console_collector() -> ConsoleCollector | None:
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

"""User-visible console messages (upstream `Env.console_logger`): records of the `pixeltable` logger marked
`user_visible`, written to stdout at the configured verbosity (`pixeltable.verbosity`: 0 warnings and above, 1 (the
default) info and above, 2 debug and above)."""

from __future__ import annotations

import logging
import sys

from pixeltable import _core
from pixeltable import exceptions as excs

_logger = logging.getLogger('pixeltable')
_console: logging.LoggerAdapter | None = None


class _ConsoleOutputHandler(logging.Handler):
    """Writes the message alone, to the current stdout."""

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        sys.stdout.write(msg if msg.endswith('\n') else msg + '\n')


class _ConsoleMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return getattr(record, 'user_visible', False)


def _map_level(verbosity: int) -> int:
    levels = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}
    if verbosity not in levels:
        raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid verbosity level: {verbosity}')
    return levels[verbosity]


def _console_logger() -> logging.LoggerAdapter:
    """Sets up the console output on first use, as upstream does when it initializes its environment."""
    global _console  # noqa: PLW0603
    if _console is None:
        value = _core.config_value('pixeltable', 'verbosity')
        verbosity = 1 if value is None else int(value)
        handler = _ConsoleOutputHandler()
        handler.setLevel(_map_level(verbosity))
        handler.addFilter(_ConsoleMessageFilter())
        if _logger.level == logging.NOTSET:
            _logger.setLevel(logging.INFO)
        _logger.propagate = False
        _logger.addHandler(handler)
        _console = logging.LoggerAdapter(_logger, extra={'user_visible': True})
    return _console


def info(msg: str) -> None:
    _console_logger().info(msg)

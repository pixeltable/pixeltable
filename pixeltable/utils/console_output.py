import logging

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
    return logging.INFO

class ConsoleOutputHandler(logging.StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        if record.msg.endswith('\n'):
            self.stream.write(record.msg)
        else:
            self.stream.write(record.msg + '\n')

class ConsoleMessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'user_visible') and record.user_visible:
            return True
        return False

class ConsoleLogger(logging.LoggerAdapter):
    def __init__(self, logger:logging.Logger):
        super().__init__(logger, extra={'user_visible' : True})


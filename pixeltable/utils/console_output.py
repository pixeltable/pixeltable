import logging

def get_level(level_str: str, default: int = logging.INFO) -> int:
    """
    Simple function to get logging level from string

    Args:
        level_str: Level string (e.g., 'INFO', 'DEBUG')
        default: Default level to use if conversion fails

    Returns:
        Logging level as integer
    """
    try:
        return getattr(logging, level_str.upper())
    except (AttributeError, TypeError):
        return default

class ConsoleOutputHandler(logging.StreamHandler):
    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        #TODO : should we prefix system console messages with PXT or similar?
        self.stream.write(record.msg)

class ConsoleMessageFilter(logging.Filter):
    def filter(self, record):
        if hasattr(record, 'user_visible') and record.user_visible:
            return True
        return False

class ConsoleLogger(logging.LoggerAdapter):
    def __init__(self, logger):
        super().__init__(logger, extra={'user_visible' : True})


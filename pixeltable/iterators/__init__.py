from .base import ComponentIterator
from .document import DocumentSplitter
from .video import FrameIterator

__default_dir = set(symbol for symbol in dir() if not symbol.startswith('_'))
__removed_symbols = {'base', 'document', 'video'}
__all__ = sorted(list(__default_dir - __removed_symbols))


def __dir__():
    return __all__

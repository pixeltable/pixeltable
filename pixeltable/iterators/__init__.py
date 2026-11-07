"""Iterators for splitting media and documents into components."""

from .audio import AudioSplitter
from .base import ComponentIterator
from .document import DocumentSplitter
from .image import TileIterator
from .string import StringSplitter
from .video import FrameIterator, VideoSplitter

__all__ = [
    'AudioSplitter',
    'ComponentIterator',
    'DocumentSplitter',
    'FrameIterator',
    'StringSplitter',
    'TileIterator',
    'VideoSplitter',
]


def __dir__() -> list[str]:
    return __all__

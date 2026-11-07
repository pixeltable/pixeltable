"""Compatibility support for custom legacy iterators."""

from .base import ComponentIterator

__all__ = ['ComponentIterator']


def __dir__() -> list[str]:
    return __all__

from typing import Any

from deprecated import deprecated

from pixeltable.func.iterator import IteratorCall
from pixeltable.iterators.base import ComponentIterator


class StringSplitter(ComponentIterator):
    @classmethod
    @deprecated(
        '`StringSplitter.create()` is deprecated; use `pixeltable.functions.string.string_splitter()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.string import string_splitter

        return string_splitter(**kwargs)

from typing import Any

from deprecated import deprecated

from pixeltable import exceptions as excs
from pixeltable.func.iterator import GeneratingFunctionCall
from pixeltable.iterators.base import ComponentIterator


class StringSplitter(ComponentIterator):
    @classmethod
    @deprecated(
        '`StringSplitter.create()` is deprecated; use `pixeltable.functions.string.string_splitter()` instead',
        version='0.5.6',
        category=excs.PixeltableDeprecationWarning,
    )
    def create(cls, **kwargs: Any) -> GeneratingFunctionCall:
        from pixeltable.functions.string import string_splitter

        return string_splitter(**kwargs)

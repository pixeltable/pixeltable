from typing import Any

from deprecated import deprecated

import pixeltable as pxt


class StringSplitter:
    @classmethod
    @deprecated(
        '`StringSplitter.create()` is deprecated; use `pixeltable.functions.string.string_splitter()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> 'pxt.func.PxtIterator':
        return pxt.functions.string.string_splitter(**kwargs)

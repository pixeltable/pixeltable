from typing import Any

from deprecated import deprecated

from pixeltable.func.iterator import IteratorCall
from pixeltable.iterators.base import ComponentIterator


class DocumentSplitter(ComponentIterator):
    @classmethod
    @deprecated(
        '`DocumentSplitter.create()` is deprecated; use `pixeltable.functions.document.document_splitter()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.document import document_splitter

        return document_splitter(**kwargs)

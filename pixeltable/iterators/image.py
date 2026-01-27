from typing import Any

from deprecated import deprecated

from pixeltable.func.iterator import IteratorCall
from pixeltable.iterators.base import ComponentIterator


class TileIterator(ComponentIterator):
    @classmethod
    @deprecated(
        '`TileIterator.create()` is deprecated; use `pixeltable.functions.image.tile_iterator()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.image import tile_iterator

        return tile_iterator(**kwargs)

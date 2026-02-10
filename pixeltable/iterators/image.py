from typing import Any

from deprecated import deprecated

from pixeltable import exceptions as excs
from pixeltable.func.iterator import GeneratingFunctionCall
from pixeltable.iterators.base import ComponentIterator


class TileIterator(ComponentIterator):
    @classmethod
    @deprecated(
        '`TileIterator.create()` is deprecated; use `pixeltable.functions.image.tile_iterator()` instead',
        version='0.5.6',
        category=excs.PixeltableDeprecationWarning,
    )
    def create(cls, **kwargs: Any) -> GeneratingFunctionCall:
        from pixeltable.functions.image import tile_iterator

        return tile_iterator(**kwargs)

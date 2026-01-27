from typing import Any

from deprecated import deprecated

import pixeltable as pxt
from pixeltable.iterators.base import ComponentIterator


class TileIterator(ComponentIterator):
    @classmethod
    @deprecated('`TileIterator.create()` is deprecated; use `pixeltable.functions.image.tile_iterator()` instead', version='0.5.6')
    def create(cls, **kwargs: Any) -> tuple[type[ComponentIterator], dict[str, Any]]:
        return pxt.functions.image.tile_iterator(**kwargs)

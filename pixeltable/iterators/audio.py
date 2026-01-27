from typing import Any

from deprecated import deprecated

from pixeltable.func.iterator import IteratorCall
from pixeltable.iterators.base import ComponentIterator


class AudioSplitter(ComponentIterator):
    @classmethod
    @deprecated(
        '`AudioSplitter.create()` is deprecated; use `pixeltable.functions.audio.audio_splitter()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.audio import audio_splitter

        return audio_splitter(**kwargs)

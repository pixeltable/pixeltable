from typing import Any

from deprecated import deprecated

from pixeltable.func.iterator import IteratorCall

from .base import ComponentIterator


class FrameIterator(ComponentIterator):
    @classmethod
    @deprecated(
        '`FrameIterator.create()` is deprecated; use `pixeltable.functions.video.frame_iterator()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> IteratorCall:
        from pixeltable.functions.video import frame_iterator

        return frame_iterator(**kwargs)


class VideoSplitter(ComponentIterator):
    @classmethod
    @deprecated(
        '`VideoSplitter.create()` is deprecated; use `pixeltable.functions.video.video_splitter()` instead',
        version='0.5.6',
    )
    def create(cls, **kwargs: Any) -> tuple[type[ComponentIterator], dict[str, Any]]:
        from pixeltable.functions.video import video_splitter

        return video_splitter(**kwargs)

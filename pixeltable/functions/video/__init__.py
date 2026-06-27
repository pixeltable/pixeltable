"""Pixeltable UDFs for VideoType."""

# ruff: noqa: F401

from pixeltable.utils.code import local_public_names

from .editing import (
    clip,
    concat_videos,
    concat_videos_agg,
    extract_audio,
    extract_frame,
    get_duration,
    get_metadata,
    make_video,
    mix_audio,
    segment_video,
    with_audio,
)
from .filters import (
    _create_drawtext_params,
    adjust_brightness,
    crop,
    fade_in,
    fade_out,
    ffmpeg_filter,
    grayscale,
    mirror_x,
    mirror_y,
    overlay_image,
    overlay_text,
    pan,
    resize,
    reverse,
    rotate,
    scroll,
    speed,
    transition,
    zoom,
)
from .iterators import (
    Frame,
    FrameAttrs,
    LegacyFrame,
    VideoSegment,
    frame_iterator,
    legacy_frame_iterator,
    video_splitter,
)
from .scene_detect import (
    SceneInfo,
    scene_detect_adaptive,
    scene_detect_content,
    scene_detect_hash,
    scene_detect_histogram,
    scene_detect_threshold,
)

__all__ = local_public_names(__name__, exclude=['editing', 'filters', 'scene_detect', 'iterators'])


def __dir__() -> list[str]:
    return __all__

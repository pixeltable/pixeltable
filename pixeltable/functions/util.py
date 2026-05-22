from typing import TypedDict

import av
import av.container
import av.stream
import PIL.Image

from pixeltable.config import Config
from pixeltable.env import Env


def resolve_torch_device(device: str, allow_mps: bool = True) -> str:
    Env.get().require_package('torch')
    import torch

    mps_enabled = Config.get().get_bool_value('enable_mps')
    if mps_enabled is None:
        mps_enabled = True  # Default to True if not set in config

    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda'
        if mps_enabled and allow_mps and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    return device


def normalize_image_mode(image: PIL.Image.Image) -> PIL.Image.Image:
    """
    Converts grayscale images to 3-channel for compatibility with models that only work with
    multichannel input.
    """
    if image.mode in ('1', 'L'):
        return image.convert('RGB')
    if image.mode == 'LA':
        return image.convert('RGBA')
    return image


class CodecContextMetadata(TypedDict, total=False):
    name: str
    codec_tag: str
    profile: str | None
    channels: int | None  # audio only
    pix_fmt: str | None  # video only


class StreamMetadata(TypedDict, total=False):
    type: str
    duration: int | None
    time_base: float | None
    duration_seconds: float | None
    frames: int
    metadata: dict[str, str]
    codec_context: CodecContextMetadata
    width: int  # video only
    height: int  # video only
    average_rate: float | None  # video only
    base_rate: float | None  # video only
    guessed_rate: float | None  # video only


class ContainerMetadata(TypedDict):
    bit_exact: bool
    bit_rate: int | None
    size: int | None
    metadata: dict[str, str]
    streams: list[StreamMetadata]


def get_metadata(path: str) -> ContainerMetadata:
    with av.open(path) as container:
        assert isinstance(container, av.container.InputContainer)
        streams_info = [__get_stream_metadata(stream) for stream in container.streams]
        result: ContainerMetadata = {
            'bit_exact': getattr(container, 'bit_exact', False),
            'bit_rate': container.bit_rate,
            'size': container.size,
            'metadata': container.metadata,
            'streams': streams_info,
        }
    return result


def __get_stream_metadata(stream: av.stream.Stream) -> StreamMetadata:
    if stream.type not in ('audio', 'video'):
        result_unsupported: StreamMetadata = {'type': stream.type}
        return result_unsupported

    codec_context = stream.codec_context
    codec_tag = codec_context.codec_tag.encode('unicode-escape').decode('utf-8')

    # Compute duration_seconds from stream-level duration.
    # We intentionally don't fall back to container.duration here because it's ambiguous —
    # it may reflect a different stream's duration (e.g. audio vs video).
    duration_seconds: float | None = None
    if stream.duration is not None and stream.time_base is not None:
        duration_seconds = float(stream.duration * stream.time_base)

    codec_context_md: CodecContextMetadata = {
        'name': codec_context.name,
        'codec_tag': codec_tag,
        'profile': codec_context.profile,
    }

    result: StreamMetadata = {
        'type': stream.type,
        'duration': stream.duration,
        'time_base': float(stream.time_base) if stream.time_base is not None else None,
        'duration_seconds': duration_seconds,
        'frames': stream.frames,
        'metadata': stream.metadata,
        'codec_context': codec_context_md,
    }

    if stream.type == 'audio':
        assert isinstance(stream.codec_context, av.AudioCodecContext)
        channels = stream.codec_context.channels
        codec_context_md['channels'] = int(channels) if channels is not None else None
    else:
        assert stream.type == 'video'
        assert isinstance(stream, av.VideoStream)
        codec_context_md['pix_fmt'] = getattr(stream.codec_context, 'pix_fmt', None)
        result['width'] = stream.width
        result['height'] = stream.height
        result['average_rate'] = float(stream.average_rate) if stream.average_rate is not None else None
        result['base_rate'] = float(stream.base_rate) if stream.base_rate is not None else None
        result['guessed_rate'] = float(stream.guessed_rate) if stream.guessed_rate is not None else None

    return result

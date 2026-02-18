"""
Shared utilities for Pixeltable functions (image normalization, media/inline_data, etc.).
"""

import base64
import mimetypes
import urllib.parse
from typing import Any, NamedTuple

import PIL.Image

from pixeltable.config import Config
from pixeltable.env import Env
from pixeltable.utils.code import local_public_names
from pixeltable.utils.http import fetch_url


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


class _ResourceInfo(NamedTuple):
    is_video: bool
    mime_type: str | None
    is_youtube: bool


def _get_resource_info(resource_str: str) -> _ResourceInfo:
    """Check if string is a video resource and return MIME type."""
    # Guess MIME type once
    guessed_type, _ = mimetypes.guess_type(resource_str, strict=False)

    # Check for youtube urls
    try:
        parsed = urllib.parse.urlparse(resource_str)
        if parsed.scheme in ('http', 'https'):
            netloc = parsed.netloc.lower().replace('www.', '')
            if (
                netloc == 'youtu.be'
                or netloc.startswith('youtube.')
                or netloc.endswith('.youtube.com')
                or netloc == 'youtube.com'
            ):
                return _ResourceInfo(is_video=True, mime_type='video/youtube', is_youtube=True)
    except (TypeError, AttributeError):
        pass

    # Check if it's a video MIME type (for non-YouTube)
    is_video = guessed_type is not None and guessed_type.lower().startswith('video/')

    return _ResourceInfo(is_video=is_video, mime_type=guessed_type, is_youtube=False)


def _media_to_inline_data_impl(element: str) -> Any:
    """
    Convert video file paths/URLs to Gemini Part format.
    YouTube URLs use file_data, other videos use inline_data, non-videos returned as-is.
    """
    resource_info = _get_resource_info(element)

    # Not a video - return as-is
    if not resource_info.is_video:
        return element

    # YouTube - use file_data with YouTube URL directly
    if resource_info.is_youtube:
        return {'file_data': {'file_uri': element}}

    # Other video file - inline as base64
    mime_type = resource_info.mime_type or 'video/mp4'  # Add fallback in case mime_type is None
    local_path = fetch_url(element, allow_local_file=True)
    data_b64 = base64.b64encode(local_path.read_bytes()).decode('utf-8')
    return {'inline_data': {'mime_type': mime_type, 'data': data_b64}}


def resolve_video_contents(element: Any) -> Any:
    """
    Recursively traverse a contents structure (list/dict) and convert only video file paths
    to Gemini Part format (inline_data / file_data). YouTube URLs and local video paths
    are converted; all other values (text, PIL Images, other media paths) are left as-is.
    """
    if isinstance(element, list):
        return [resolve_video_contents(v) for v in element]
    if isinstance(element, dict):
        if 'inline_data' in element or 'file_data' in element:
            return element
        return {k: resolve_video_contents(v) for k, v in element.items()}
    if isinstance(element, str):
        return _media_to_inline_data_impl(element)
    return element


__all__ = local_public_names(__name__)

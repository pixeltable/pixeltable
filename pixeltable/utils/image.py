import base64
import threading
from io import BytesIO
from typing import Any

import PIL.Image

_WEBP_MODES = frozenset(['RGBA', 'RGBa'])
_JPEG_MODES = frozenset(['L', 'RGB', 'RGBX', 'CMYK', 'YCbCr'])
# PNG round-trips these modes, including their transparency; it would store mode I as I;16
PNG_MODES = frozenset(['1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I;16'])


def default_format(img: PIL.Image.Image) -> str:
    """Return a format that can encode img in its mode; RGBX and YCbCr images come back as RGB, RGBa images as RGBA."""
    if img.mode in _WEBP_MODES:
        # WebP drops a fully opaque alpha channel, so such an image comes back as RGB
        return 'webp'
    if img.mode in _JPEG_MODES and not img.has_transparency_data:
        return 'jpeg'
    if img.mode in PNG_MODES:
        return 'png'
    return 'tiff'


def load(path: str) -> PIL.Image.Image:
    """Return the decoded image stored in the file at path."""
    img = PIL.Image.open(path)
    # PIL.Image.open() reads only the file header. load() decodes the pixels on the calling thread, so that no later
    # reader of this shared image starts the decode.
    img.load()
    return img


def save(image: PIL.Image.Image, fp: Any, format: str | None = None) -> None:
    """Save image to fp, which is a path or a file object. Without a format, Pillow derives it from the path.

    Image.save() keeps its encoder settings on the image while it runs. Two threads that save the same image at the
    same time would use each other's settings, so every save of an image holds the lock of that image.
    """
    # dict.setdefault() is atomic, so two threads get the same lock
    save_lock = image.__dict__.setdefault('_pxt_save_lock', threading.Lock())
    with save_lock:
        image.save(fp, format=format)


def to_base64(image: PIL.Image.Image | str, format: str | None = None) -> str:
    return base64.b64encode(to_bytes(image, format)).decode('utf-8')


def to_bytes(image: PIL.Image.Image | str, format: str | None = None) -> bytes:
    if isinstance(image, str):
        with open(image, 'rb') as f:
            return f.read()
    if image.width == 0 or image.height == 0:
        # Pillow's error for an empty image depends on the format and version, e.g. MemoryError for WebP in Pillow 12;
        # we want a more informative error here
        raise ValueError('cannot write empty image')
    buffer = BytesIO()
    save(image, buffer, format or default_format(image))
    return buffer.getvalue()

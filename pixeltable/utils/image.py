import base64
from io import BytesIO

import PIL.Image

_WEBP_MODES = frozenset(['RGBA', 'RGBa'])
_JPEG_MODES = frozenset(['L', 'RGB', 'RGBX', 'CMYK', 'YCbCr'])
# PNG round-trips these modes, including their transparency; it would store mode I as I;16
PNG_MODES = frozenset(['1', 'L', 'LA', 'P', 'RGB', 'RGBA', 'I;16'])


def default_format(img: PIL.Image.Image) -> str:
    """Return a format that can encode img in its mode, or as RGB for modes RGBX, YCbCr and RGBa."""
    if img.mode in _WEBP_MODES:
        # WebP drops a fully opaque alpha channel, so such an RGBA image comes back as RGB
        return 'webp'
    if img.mode in _JPEG_MODES and not img.has_transparency_data:
        return 'jpeg'
    if img.mode in PNG_MODES:
        return 'png'
    return 'tiff'


def to_base64(image: PIL.Image.Image | str, format: str | None = None) -> str:
    return base64.b64encode(to_bytes(image, format)).decode('utf-8')


def to_bytes(image: PIL.Image.Image | str, format: str | None = None) -> bytes:
    if isinstance(image, str):
        with open(image, 'rb') as f:
            return f.read()
    buffer = BytesIO()
    image.save(buffer, format=format or default_format(image))
    return buffer.getvalue()

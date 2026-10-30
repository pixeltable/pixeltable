import base64
from io import BytesIO

import PIL.Image

_JPEG_MODES = frozenset(['1', 'L', 'RGB', 'RGBX', 'CMYK', 'YCbCr'])
_PNG_MODES = frozenset(['P', 'I;16'])


def default_format(img: PIL.Image.Image) -> str:
    """Return a format that can encode img in its mode."""
    if img.has_transparency_data:
        return 'webp'
    if img.mode in _JPEG_MODES:
        return 'jpeg'
    # PNG and TIFF keep the mode; PNG would store mode I as I;16
    if img.mode in _PNG_MODES:
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

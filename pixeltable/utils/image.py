import base64
from io import BytesIO

import PIL.Image


def default_format(img: PIL.Image.Image) -> str:
    # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
    # In that case, use WebP instead.
    return 'webp' if img.has_transparency_data else 'jpeg'


def to_base64(image: PIL.Image.Image | str, format: str | None = None) -> str:
    return base64.b64encode(to_bytes(image, format)).decode('utf-8')


def to_bytes(image: PIL.Image.Image | str, format: str | None = None) -> bytes:
    if isinstance(image, str):
        with open(image, 'rb') as f:
            return f.read()
    buffer = BytesIO()
    image.save(buffer, format=format or default_format(image))
    return buffer.getvalue()

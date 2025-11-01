import base64
from io import BytesIO

import PIL.Image


def default_format(img: PIL.Image.Image) -> str:
    # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
    # In that case, use WebP instead.
    return 'webp' if img.has_transparency_data else 'jpeg'


def to_base64(image: PIL.Image.Image, format: str | None = None) -> str:
    buffer = BytesIO()
    image.save(buffer, format=format or image.format)
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')

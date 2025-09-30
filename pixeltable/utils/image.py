import PIL.Image


def default_format(img: PIL.Image.Image) -> str:
    # Default to JPEG unless the image has a transparency layer (which isn't supported by JPEG).
    # In that case, use WebP instead.
    return 'webp' if img.has_transparency_data else 'jpeg'

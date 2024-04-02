import base64

import PIL.Image

from pixeltable.type_system import ImageType, StringType
import pixeltable.func as func


@func.udf
def b64_encode(img: PIL.Image.Image, image_format: str = 'png') -> str:
    # Encode this image as a b64-encoded png.
    import io
    bytes_arr = io.BytesIO()
    img.save(bytes_arr, format=image_format)
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    return b64_bytes.decode('utf-8')

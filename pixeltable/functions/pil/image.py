import base64
import sys
from typing import Dict, Any, Tuple, Optional

import PIL.Image

from pixeltable.type_system import FloatType, ImageType, IntType, ArrayType, ColumnType, StringType, JsonType
import pixeltable.func as func


alpha_composite = func.make_library_function(
    ImageType(), [ImageType(), ImageType()], 'PIL.Image', 'alpha_composite')
blend = func.make_library_function(ImageType(), [ImageType(), ImageType(), FloatType()], 'PIL.Image', 'blend')
composite = func.make_library_function(
    ImageType(), [ImageType(), ImageType(), ImageType()], 'PIL.Image', 'composite')


# methods of PIL.Image.Image

def _b64_encode(self: PIL.Image.Image) -> str:
    # Encode this image as a b64-encoded png.
    import io
    bytes_arr = io.BytesIO()
    self.save(bytes_arr, format='png')
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    return b64_bytes.decode('utf-8')
b64_encode = func.make_library_function(StringType(), [ImageType()], __name__, '_b64_encode')

def _caller_return_type(bound_args: Optional[Dict[str, Any]]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    return bound_args['self'].col_type

# Image.convert()
def _convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    return self.convert(mode)
def _convert_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    assert 'self' in bound_args
    assert 'mode' in bound_args
    img_type = bound_args['self'].col_type
    return ImageType(size=img_type.size, mode=bound_args['mode'])
convert = func.make_library_function(
    _convert_return_type, [ImageType(), StringType()], __name__, '_convert')

# Image.crop()
def _crop(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> PIL.Image.Image:
    return self.crop(box)
def _crop_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    img_type = bound_args['self'].col_type
    box = bound_args['box']
    if isinstance(box, list) and all(isinstance(x, int) for x in box):
        return ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=img_type.mode)
    return ImageType()  # we can't compute the size statically
crop = func.make_library_function(
    _crop_return_type, [ImageType(), ArrayType((4,), dtype=IntType())], __name__, '_crop')

# Image.entropy()
def _entropy(self: PIL.Image.Image) -> float:
    return self.entropy()
entropy = func.make_library_function(FloatType(), [ImageType()], __name__, '_entropy')

# Image.getchannel()
def _getchannel_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    img_type = bound_args['self'].col_type
    return ImageType(size=img_type.size, mode='L')
getchannel = func.make_library_function(
    _getchannel_return_type, [ImageType(), IntType()], 'PIL.Image', 'Image.getchannel')

# Image.histogram()
def _histogram(self: PIL.Image.Image) -> JsonType():
    return self.histogram()
histogram = func.make_library_function(JsonType(), [ImageType()], __name__, '_histogram')

# Image.resize()
def _resize(self: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return self.resize(size)
def resize_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    assert 'size' in bound_args
    return ImageType(size=bound_args['size'])
resize = func.make_library_function(
    resize_return_type, [ImageType(), ArrayType((2, ), dtype=IntType())], __name__, '_resize')

# Image.rotate()
def _rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    return self.rotate(angle)
rotate = func.make_library_function(
    _caller_return_type, [ImageType(), IntType()], __name__, '_rotate')

# Image.transform()
def _transform(self: PIL.Image.Image, size: Tuple[int, int], method: int) -> PIL.Image.Image:
    return self.transform(size, method)
transform = func.make_library_function(
    _caller_return_type, [ImageType(), ArrayType((2,), dtype=IntType()), IntType()], __name__, '_transform')

# Image.filter()
# TODO: what should the filter type be?
#filter = make_library_function(_caller_return_type(), [ImageType(), ImageType()], 'PIL.Image', 'Image.filter')

effect_spread = func.make_library_function(
    _caller_return_type, [ImageType(), FloatType()], 'PIL.Image', 'Image.effect_spread')
getbbox = func.make_library_function(
    ArrayType((4,), dtype=IntType()), [ImageType()], 'PIL.Image', 'Image.getbbox')
getcolors = func.make_library_function(
    JsonType(), [ImageType(), IntType()], 'PIL.Image', 'Image.getcolors')
getextrema = func.make_library_function(JsonType(), [ImageType()], 'PIL.Image', 'Image.getextrema')
getpalette = \
    func.make_library_function(JsonType(), [ImageType(), StringType()], 'PIL.Image', 'Image.getpalette')
getpixel = func.make_library_function(
    JsonType(), [ImageType(), ArrayType((2,), dtype=IntType())], 'PIL.Image', 'Image.getpixel')
getprojection = func.make_library_function(JsonType(), [ImageType()], 'PIL.Image', 'Image.getprojection')
quantize = func.make_library_function(
    ImageType(), [ImageType(), IntType(), IntType(nullable=True), IntType(), IntType(nullable=True), IntType()],
    'PIL.Image', 'Image.quantize')
reduce = \
    func.make_library_function(ImageType(), [ImageType(), IntType(), JsonType()], 'PIL.Image', 'Image.reduce')
transpose = func.make_library_function(
    _caller_return_type, [ImageType(), IntType()], 'PIL.Image', 'Image.transpose')

func.FunctionRegistry.get().register_module(sys.modules[__name__])

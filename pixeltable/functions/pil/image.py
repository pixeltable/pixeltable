from typing import Dict, Any, Tuple, Optional

import PIL.Image
from PIL.Image import Dither, Resampling

from pixeltable.type_system import FloatType, ImageType, IntType, ArrayType, ColumnType, StringType, JsonType, BoolType
import pixeltable.func as func


def _caller_return_type(self: PIL.Image.Image) -> ColumnType:
    return self.col_type

@func.udf(
    py_fn=PIL.Image.alpha_composite, return_type=ImageType(), param_types=[ImageType(), ImageType()])
def alpha_composite(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    pass
@func.udf(
    py_fn=PIL.Image.blend, return_type=ImageType(), param_types=[ImageType(), ImageType(), FloatType()])
def blend(im1: PIL.Image.Image, im2: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    pass
@func.udf(
    py_fn=PIL.Image.composite, return_type=ImageType(), param_types=[ImageType(), ImageType(), ImageType()])
def composite(image1: PIL.Image.Image, image2: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    pass


# PIL.Image.Image methods

# Image.convert()
def _convert_return_type(self: PIL.Image.Image, mode: str) -> ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ImageType(size=input_type.size, mode=mode, nullable=input_type.nullable)
@func.udf(call_return_type=_convert_return_type, param_types=[ImageType(), StringType()])
def convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    return self.convert(mode)

# Image.crop()
def _crop_return_type(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    if isinstance(box, list) and all(isinstance(x, int) for x in box):
        return ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=input_type.mode, nullable=input_type.nullable)
    return ImageType(mode=input_type.mode, nullable=input_type.nullable)  # we can't compute the size statically
@func.udf(
    py_fn=PIL.Image.Image.crop, call_return_type=_crop_return_type,
    param_types=[ImageType(), ArrayType((4,), dtype=IntType())])
def crop(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> PIL.Image.Image:
    pass

# Image.getchannel()
def _getchannel_return_type(self: PIL.Image.Image) -> ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ImageType(size=input_type.size, mode='L', nullable=input_type.nullable)
@func.udf(
    py_fn=PIL.Image.Image.getchannel, call_return_type=_getchannel_return_type, param_types=[ImageType(), IntType()])
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    pass

# Image.resize()
def _resize_return_type(self: PIL.Image.Image, size: Tuple[int, int]) -> ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ImageType(size=size, mode=input_type.mode, nullable=input_type.nullable)
@func.udf(call_return_type=_resize_return_type, param_types=[ImageType(), ArrayType((2, ), dtype=IntType())])
def resize(self: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return self.resize(size)

# Image.rotate()
@func.udf(call_return_type=_caller_return_type, param_types=[ImageType(), IntType()])
def rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    return self.rotate(angle)

@func.udf(
    py_fn=PIL.Image.Image.effect_spread, call_return_type=_caller_return_type, param_types=[ImageType(), IntType()])
def effect_spread(self: PIL.Image.Image, distance: int) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.entropy, return_type=FloatType(), param_types=[ImageType(), ImageType(), JsonType()])
def entropy(self: PIL.Image.Image, mask: PIL.Image.Image, extrema: Optional[list] = None) -> float:
    pass

@func.udf(py_fn=PIL.Image.Image.getbands, return_type=JsonType(), param_types=[ImageType()])
def getbands(self: PIL.Image.Image) -> Tuple[str]:
    pass

@func.udf(py_fn=PIL.Image.Image.getbbox, return_type=JsonType(), param_types=[ImageType()])
def getbbox(self: PIL.Image.Image) -> Tuple[int, int, int, int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.getcolors, return_type=JsonType(), param_types=[ImageType(), IntType()])
def getcolors(self: PIL.Image.Image, maxcolors: int) -> Tuple[Tuple[int, int, int], int]:
    pass

@func.udf(py_fn=PIL.Image.Image.getextrema, return_type=JsonType(), param_types=[ImageType()])
def getextrema(self: PIL.Image.Image) -> Tuple[int, int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.getpalette, return_type=JsonType(), param_types=[ImageType(), StringType()])
def getpalette(self: PIL.Image.Image, mode: Optional[str] = None) -> Tuple[int]:
    pass

@func.udf(
    return_type=JsonType(), param_types=[ImageType(), ArrayType((2,), dtype=IntType())])
def getpixel(self: PIL.Image.Image, xy: tuple[int, int]) -> Tuple[int]:
    # `xy` will be a list; `tuple(xy)` is necessary for pillow 9 compatibility
    return self.getpixel(tuple(xy))

@func.udf(
    py_fn=PIL.Image.Image.getprojection, return_type=JsonType(), param_types=[ImageType()])
def getprojection(self: PIL.Image.Image) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.histogram, return_type=JsonType(), param_types=[ImageType(), ImageType(), JsonType()])
def histogram(self: PIL.Image.Image, mask: PIL.Image.Image, extrema: Optional[list] = None) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.quantize, return_type=ImageType(),
    param_types=[ImageType(), IntType(), IntType(nullable=True), IntType(), IntType(nullable=True), IntType()])
def quantize(
        self: PIL.Image.Image, colors: int = 256, method: Optional[int] = None, kmeans: int = 0,
        palette: Optional[int] = None, dither: int = Dither.FLOYDSTEINBERG) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.reduce, return_type=ImageType(), param_types=[ImageType(), IntType(), JsonType()])
def reduce(self: PIL.Image.Image, factor: int, box: Optional[Tuple[int]]) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.transpose, call_return_type=_caller_return_type, param_types=[ImageType(), IntType()])
def transpose(self: PIL.Image.Image, method: int) -> PIL.Image.Image:
    pass

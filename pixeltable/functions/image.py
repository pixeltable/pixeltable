import base64
from typing import Optional, Tuple

import PIL.Image

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.utils.code import local_public_names

@func.udf
def b64_encode(img: PIL.Image.Image, image_format: str = 'png') -> str:
    # Encode this image as a b64-encoded png.
    import io
    bytes_arr = io.BytesIO()
    img.save(bytes_arr, format=image_format)
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    return b64_bytes.decode('utf-8')


@func.udf(
    py_fn=PIL.Image.alpha_composite, return_type=ts.ImageType(), param_types=[ts.ImageType(), ts.ImageType()])
def alpha_composite(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    pass
@func.udf(
    py_fn=PIL.Image.blend, return_type=ts.ImageType(), param_types=[ts.ImageType(), ts.ImageType(), ts.FloatType()])
def blend(im1: PIL.Image.Image, im2: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    pass
@func.udf(
    py_fn=PIL.Image.composite, return_type=ts.ImageType(), param_types=[ts.ImageType(), ts.ImageType(), ts.ImageType()])
def composite(image1: PIL.Image.Image, image2: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    pass


# PIL.Image.Image methods

# Image.convert()
@func.udf(param_types=[ts.ImageType(), ts.StringType()])
def convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    return self.convert(mode)


@convert.conditional_return_type
def _(self: PIL.Image.Image, mode: str) -> ts.ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ts.ImageType(size=input_type.size, mode=mode, nullable=input_type.nullable)


# Image.crop()
@func.udf(
    py_fn=PIL.Image.Image.crop,
    param_types=[ts.ImageType(), ts.ArrayType((4,), dtype=ts.IntType())])
def crop(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> PIL.Image.Image:
    pass

@crop.conditional_return_type
def _(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    if isinstance(box, list) and all(isinstance(x, int) for x in box):
        return ts.ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=input_type.mode, nullable=input_type.nullable)
    return ts.ImageType(mode=input_type.mode, nullable=input_type.nullable)  # we can't compute the size statically

# Image.getchannel()
@func.udf(py_fn=PIL.Image.Image.getchannel, param_types=[ts.ImageType(), ts.IntType()])
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    pass

@getchannel.conditional_return_type
def _(self: PIL.Image.Image) -> ts.ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ts.ImageType(size=input_type.size, mode='L', nullable=input_type.nullable)


# Image.resize()
@func.udf(param_types=[ts.ImageType(), ts.ArrayType((2, ), dtype=ts.IntType())])
def resize(self: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return self.resize(size)

@resize.conditional_return_type
def _(self: PIL.Image.Image, size: Tuple[int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert input_type.is_image_type()
    return ts.ImageType(size=size, mode=input_type.mode, nullable=input_type.nullable)

# Image.rotate()
@func.udf(param_types=[ts.ImageType(), ts.IntType()])
def rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    return self.rotate(angle)

@func.udf(py_fn=PIL.Image.Image.effect_spread, param_types=[ts.ImageType(), ts.IntType()])
def effect_spread(self: PIL.Image.Image, distance: int) -> PIL.Image.Image:
    pass

@func.udf(py_fn=PIL.Image.Image.transpose, param_types=[ts.ImageType(), ts.IntType()])
def transpose(self: PIL.Image.Image, method: int) -> PIL.Image.Image:
    pass

@rotate.conditional_return_type
@effect_spread.conditional_return_type
@transpose.conditional_return_type
def _(self: PIL.Image.Image) -> ts.ColumnType:
    return self.col_type

@func.udf(
    py_fn=PIL.Image.Image.entropy, return_type=ts.FloatType(), param_types=[ts.ImageType(), ts.ImageType(), ts.JsonType()])
def entropy(self: PIL.Image.Image, mask: PIL.Image.Image, extrema: Optional[list] = None) -> float:
    pass

@func.udf(py_fn=PIL.Image.Image.getbands, return_type=ts.JsonType(), param_types=[ts.ImageType()])
def getbands(self: PIL.Image.Image) -> Tuple[str]:
    pass

@func.udf(py_fn=PIL.Image.Image.getbbox, return_type=ts.JsonType(), param_types=[ts.ImageType()])
def getbbox(self: PIL.Image.Image) -> Tuple[int, int, int, int]:
    pass

@func.udf(py_fn=PIL.Image.Image.getcolors, return_type=ts.JsonType(), param_types=[ts.ImageType(), ts.IntType()])
def getcolors(self: PIL.Image.Image, maxcolors: int) -> Tuple[Tuple[int, int, int], int]:
    pass

@func.udf(py_fn=PIL.Image.Image.getextrema, return_type=ts.JsonType(), param_types=[ts.ImageType()])
def getextrema(self: PIL.Image.Image) -> Tuple[int, int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.getpalette, return_type=ts.JsonType(), param_types=[ts.ImageType(), ts.StringType()])
def getpalette(self: PIL.Image.Image, mode: Optional[str] = None) -> Tuple[int]:
    pass

@func.udf(
    return_type=ts.JsonType(), param_types=[ts.ImageType(), ts.ArrayType((2,), dtype=ts.IntType())])
def getpixel(self: PIL.Image.Image, xy: tuple[int, int]) -> Tuple[int]:
    # `xy` will be a list; `tuple(xy)` is necessary for pillow 9 compatibility
    return self.getpixel(tuple(xy))

@func.udf(py_fn=PIL.Image.Image.getprojection, return_type=ts.JsonType(), param_types=[ts.ImageType()])
def getprojection(self: PIL.Image.Image) -> Tuple[int]:
    pass

@func.udf(py_fn=PIL.Image.Image.histogram, return_type=ts.JsonType(), param_types=[ts.ImageType(), ts.ImageType(), ts.JsonType()])
def histogram(self: PIL.Image.Image, mask: PIL.Image.Image, extrema: Optional[list] = None) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.quantize, return_type=ts.ImageType(),
    param_types=[ts.ImageType(), ts.IntType(), ts.IntType(nullable=True), ts.IntType(), ts.IntType(nullable=True), ts.IntType()])
def quantize(
        self: PIL.Image.Image, colors: int = 256, method: Optional[int] = None, kmeans: int = 0,
        palette: Optional[int] = None, dither: int = PIL.Image.Dither.FLOYDSTEINBERG) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.reduce, return_type=ts.ImageType(), param_types=[ts.ImageType(), ts.IntType(), ts.JsonType()])
def reduce(self: PIL.Image.Image, factor: int, box: Optional[Tuple[int]]) -> PIL.Image.Image:
    pass


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

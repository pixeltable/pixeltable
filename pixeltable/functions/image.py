import base64
from typing import Optional, Tuple

import PIL.Image

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable.utils.code import local_public_names
from pixeltable.exprs import Expr


@func.udf(is_method=True)
def b64_encode(img: PIL.Image.Image, image_format: str = 'png') -> str:
    # Encode this image as a b64-encoded png.
    import io

    bytes_arr = io.BytesIO()
    img.save(bytes_arr, format=image_format)
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    return b64_bytes.decode('utf-8')


@func.udf(substitute_fn=PIL.Image.alpha_composite, is_method=True)
def alpha_composite(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    pass


@func.udf(substitute_fn=PIL.Image.blend, is_method=True)
def blend(im1: PIL.Image.Image, im2: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    pass


@func.udf(substitute_fn=PIL.Image.composite, is_method=True)
def composite(image1: PIL.Image.Image, image2: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    pass


# PIL.Image.Image methods


# Image.convert()
@func.udf(is_method=True)
def convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    return self.convert(mode)


@convert.conditional_return_type
def _(self: Expr, mode: str) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=input_type.size, mode=mode, nullable=input_type.nullable)


# Image.crop()
@func.udf(substitute_fn=PIL.Image.Image.crop, param_types=[ts.ImageType(), ts.ArrayType((4,), dtype=ts.IntType())], is_method=True)
def crop(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> PIL.Image.Image:
    pass


@crop.conditional_return_type
def _(self: Expr, box: Tuple[int, int, int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    if isinstance(box, list) and all(isinstance(x, int) for x in box):
        return ts.ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=input_type.mode, nullable=input_type.nullable)
    return ts.ImageType(mode=input_type.mode, nullable=input_type.nullable)  # we can't compute the size statically


# Image.getchannel()
@func.udf(substitute_fn=PIL.Image.Image.getchannel, is_method=True)
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    pass


@getchannel.conditional_return_type
def _(self: Expr) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=input_type.size, mode='L', nullable=input_type.nullable)


# Image.resize()
@func.udf(param_types=[ts.ImageType(), ts.ArrayType((2,), dtype=ts.IntType())], is_method=True)
def resize(self: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return self.resize(size)


@resize.conditional_return_type
def _(self: Expr, size: Tuple[int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=size, mode=input_type.mode, nullable=input_type.nullable)


# Image.rotate()
@func.udf(is_method=True)
def rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    return self.rotate(angle)


@func.udf(substitute_fn=PIL.Image.Image.effect_spread, is_method=True)
def effect_spread(self: PIL.Image.Image, distance: int) -> PIL.Image.Image:
    pass


@func.udf(substitute_fn=PIL.Image.Image.transpose, is_method=True)
def transpose(self: PIL.Image.Image, method: int) -> PIL.Image.Image:
    pass


@rotate.conditional_return_type
@effect_spread.conditional_return_type
@transpose.conditional_return_type
def _(self: Expr) -> ts.ColumnType:
    return self.col_type


@func.udf(substitute_fn=PIL.Image.Image.entropy, is_method=True)
def entropy(self: PIL.Image.Image, mask: Optional[PIL.Image.Image] = None, extrema: Optional[list] = None) -> float:
    pass


@func.udf(substitute_fn=PIL.Image.Image.getbands, is_method=True)
def getbands(self: PIL.Image.Image) -> Tuple[str]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.getbbox, is_method=True)
def getbbox(self: PIL.Image.Image) -> Tuple[int, int, int, int]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.getcolors, is_method=True)
def getcolors(self: PIL.Image.Image, maxcolors: int) -> Tuple[Tuple[int, int, int], int]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.getextrema, is_method=True)
def getextrema(self: PIL.Image.Image) -> Tuple[int, int]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.getpalette, is_method=True)
def getpalette(self: PIL.Image.Image, mode: Optional[str] = None) -> Tuple[int]:
    pass


@func.udf(param_types=[ts.ImageType(), ts.ArrayType((2,), dtype=ts.IntType())], is_method=True)
def getpixel(self: PIL.Image.Image, xy: tuple[int, int]) -> Tuple[int]:
    # `xy` will be a list; `tuple(xy)` is necessary for pillow 9 compatibility
    return self.getpixel(tuple(xy))


@func.udf(substitute_fn=PIL.Image.Image.getprojection, is_method=True)
def getprojection(self: PIL.Image.Image) -> Tuple[int]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.histogram, is_method=True)
def histogram(self: PIL.Image.Image, mask: PIL.Image.Image, extrema: Optional[list] = None) -> Tuple[int]:
    pass


@func.udf(substitute_fn=PIL.Image.Image.quantize, is_method=True)
def quantize(
    self: PIL.Image.Image,
    colors: int = 256,
    method: Optional[int] = None,
    kmeans: int = 0,
    palette: Optional[int] = None,
    dither: int = PIL.Image.Dither.FLOYDSTEINBERG,
) -> PIL.Image.Image:
    pass


@func.udf(substitute_fn=PIL.Image.Image.reduce, is_method=True)
def reduce(self: PIL.Image.Image, factor: int, box: Optional[Tuple[int]] = None) -> PIL.Image.Image:
    pass


@func.udf(is_property=True)
def width(self: PIL.Image.Image) -> int:
    return self.width


@func.udf(is_property=True)
def height(self: PIL.Image.Image) -> int:
    return self.height


@func.udf(is_property=True)
def mode(self: PIL.Image.Image) -> str:
    return self.mode


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

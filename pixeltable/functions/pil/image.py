from typing import Dict, Any, Tuple, Optional

import PIL.Image

from pixeltable.type_system import FloatType, ImageType, IntType, ArrayType, ColumnType, StringType, JsonType, BoolType
import pixeltable.func as func


def _caller_return_type(bound_args: Optional[Dict[str, Any]]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    return bound_args['self'].col_type

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
def _convert_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    assert 'self' in bound_args
    assert 'mode' in bound_args
    img_type = bound_args['self'].col_type
    return ImageType(size=img_type.size, mode=bound_args['mode'])
@func.udf(return_type=_convert_return_type, param_types=[ImageType(), StringType()])
def convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    return self.convert(mode)

# Image.crop()
def _crop_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    img_type = bound_args['self'].col_type
    box = bound_args['box']
    if isinstance(box, list) and all(isinstance(x, int) for x in box):
        return ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=img_type.mode)
    return ImageType()  # we can't compute the size statically
@func.udf(
    py_fn=PIL.Image.Image.crop, return_type=_crop_return_type,
    param_types=[ImageType(), ArrayType((4,), dtype=IntType())])
def crop(self: PIL.Image.Image, box: Tuple[int, int, int, int]) -> PIL.Image.Image:
    pass

# Image.getchannel()
def _getchannel_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    img_type = bound_args['self'].col_type
    return ImageType(size=img_type.size, mode='L')
@func.udf(
    py_fn=PIL.Image.Image.getchannel, return_type=_getchannel_return_type, param_types=[ImageType(), IntType()])
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    pass

# Image.resize()
def resize_return_type(bound_args: Dict[str, Any]) -> ColumnType:
    if bound_args is None:
        return ImageType()
    assert 'size' in bound_args
    return ImageType(size=bound_args['size'])
@func.udf(return_type=resize_return_type, param_types=[ImageType(), ArrayType((2, ), dtype=IntType())])
def resize(self: PIL.Image.Image, size: Tuple[int, int]) -> PIL.Image.Image:
    return self.resize(size)

# Image.rotate()
@func.udf(return_type=ImageType(), param_types=[ImageType(), IntType()])
def rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    return self.rotate(angle)

# Image.transform()
@func.udf(return_type= _caller_return_type, param_types=[ImageType(), ArrayType((2,), dtype=IntType()), IntType()])
def transform(self: PIL.Image.Image, size: Tuple[int, int], method: int) -> PIL.Image.Image:
    return self.transform(size, method)

@func.udf(
    py_fn=PIL.Image.Image.effect_spread, return_type=_caller_return_type, param_types=[ImageType(), FloatType()])
def effect_spread(self: PIL.Image.Image, distance: float) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.entropy, return_type=FloatType(), param_types=[ImageType(), ImageType(), JsonType()])
def entropy(self: PIL.Image.Image, mask: PIL.Image.Image, histogram: Dict) -> float:
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
def getpalette(self: PIL.Image.Image, mode: str) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.getpixel, return_type=JsonType(), param_types=[ImageType(), ArrayType((2,), dtype=IntType())])
def getpixel(self: PIL.Image.Image, xy: Tuple[int, int]) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.getprojection, return_type=JsonType(), param_types=[ImageType()])
def getprojection(self: PIL.Image.Image) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.histogram, return_type=JsonType(), param_types=[ImageType(), ImageType(), JsonType()])
def histogram(self: PIL.Image.Image, mask: PIL.Image.Image, histogram: Dict) -> Tuple[int]:
    pass

@func.udf(
    py_fn=PIL.Image.Image.quantize, return_type=ImageType(),
    param_types=[ImageType(), IntType(), IntType(nullable=True), IntType(), IntType(nullable=True), IntType()])
def quantize(
        self: PIL.Image.Image, colors: int, method: int, kmeans: int, palette: int, dither: int) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.reduce, return_type=ImageType(), param_types=[ImageType(), IntType(), JsonType()])
def reduce(self: PIL.Image.Image, factor: int, filter: Tuple[int]) -> PIL.Image.Image:
    pass

@func.udf(
    py_fn=PIL.Image.Image.transpose, return_type=_caller_return_type, param_types=[ImageType(), IntType()])
def transpose(self: PIL.Image.Image, method: int) -> PIL.Image.Image:
    pass

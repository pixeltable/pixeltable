"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for `ImageType`.

Example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.img_col.convert('L')).collect()
```
"""

import base64
from typing import Optional

import PIL.Image

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.exprs import Expr
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def b64_encode(img: PIL.Image.Image, image_format: str = 'png') -> str:
    """
    Convert image to a base64-encoded string.

    Args:
        img: image
        image_format: image format [supported by PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats)
    """
    import io

    bytes_arr = io.BytesIO()
    img.save(bytes_arr, format=image_format)
    b64_bytes = base64.b64encode(bytes_arr.getvalue())
    return b64_bytes.decode('utf-8')


@pxt.udf(substitute_fn=PIL.Image.alpha_composite, is_method=True)
def alpha_composite(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    """
    Alpha composite `im2` over `im1`.

    Equivalent to [`PIL.Image.alpha_composite()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.alpha_composite)
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.blend, is_method=True)
def blend(im1: PIL.Image.Image, im2: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    """
    Return a new image by interpolating between two input images, using a constant alpha.

    Equivalent to [`PIL.Image.blend()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.blend)
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.composite, is_method=True)
def composite(image1: PIL.Image.Image, image2: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    """
    Return a composite image by blending two images using a mask.

    Equivalent to [`PIL.Image.composite()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.composite)
    """
    pass


# PIL.Image.Image methods


# Image.convert()
@pxt.udf(is_method=True)
def convert(self: PIL.Image.Image, mode: str) -> PIL.Image.Image:
    """
    Convert the image to a different mode.

    Equivalent to
    [`PIL.Image.Image.convert()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert).

    Args:
        mode: The mode to convert to. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes)
            for a list of supported modes.
    """
    return self.convert(mode)


@convert.conditional_return_type
def _(self: Expr, mode: str) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=input_type.size, mode=mode, nullable=input_type.nullable)


# Image.crop()
@pxt.udf(substitute_fn=PIL.Image.Image.crop, is_method=True)
def crop(self: PIL.Image.Image, box: tuple[int, int, int, int]) -> PIL.Image.Image:
    """
    Return a rectangular region from the image. The box is a 4-tuple defining the left, upper, right, and lower pixel
    coordinates.

    Equivalent to
    [`PIL.Image.Image.crop()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop)
    """
    pass


@crop.conditional_return_type
def _(self: Expr, box: tuple[int, int, int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    if (isinstance(box, (list, tuple))) and len(box) == 4 and all(isinstance(x, int) for x in box):
        return ts.ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=input_type.mode, nullable=input_type.nullable)
    return ts.ImageType(mode=input_type.mode, nullable=input_type.nullable)  # we can't compute the size statically


# Image.getchannel()
@pxt.udf(substitute_fn=PIL.Image.Image.getchannel, is_method=True)
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    """
    Return an L-mode image containing a single channel of the original image.

    Equivalent to
    [`PIL.Image.Image.getchannel()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getchannel)

    Args:
        channel: The channel to extract. This is a 0-based index.
    """
    pass


@getchannel.conditional_return_type
def _(self: Expr) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=input_type.size, mode='L', nullable=input_type.nullable)


@pxt.udf(is_method=True)
def get_metadata(self: PIL.Image.Image) -> dict:
    """
    Return metadata for the image.
    """
    return {
        'width': self.width,
        'height': self.height,
        'mode': self.mode,
        'bits': getattr(self, 'bits', None),
        'format': self.format,
        'palette': self.palette,
    }


# Image.point()
@pxt.udf(is_method=True)
def point(self: PIL.Image.Image, lut: list[int], mode: Optional[str] = None) -> PIL.Image.Image:
    """
    Map image pixels through a lookup table.

    Equivalent to
    [`PIL.Image.Image.point()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.point)

    Args:
        lut: A lookup table.
    """
    return self.point(lut, mode=mode)


# Image.resize()
@pxt.udf(is_method=True)
def resize(self: PIL.Image.Image, size: tuple[int, int]) -> PIL.Image.Image:
    """
    Return a resized copy of the image. The size parameter is a tuple containing the width and height of the new image.

    Equivalent to
    [`PIL.Image.Image.resize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize)
    """
    return self.resize(tuple(size))  # type: ignore[arg-type]


@resize.conditional_return_type
def _(self: Expr, size: tuple[int, int]) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    return ts.ImageType(size=size, mode=input_type.mode, nullable=input_type.nullable)


# Image.rotate()
@pxt.udf(is_method=True)
def rotate(self: PIL.Image.Image, angle: int) -> PIL.Image.Image:
    """
    Return a copy of the image rotated by the given angle.

    Equivalent to
    [`PIL.Image.Image.rotate()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.rotate)

    Args:
        angle: The angle to rotate the image, in degrees. Positive angles are counter-clockwise.
    """
    return self.rotate(angle)


@pxt.udf(substitute_fn=PIL.Image.Image.effect_spread, is_method=True)
def effect_spread(self: PIL.Image.Image, distance: int) -> PIL.Image.Image:
    """
    Randomly spread pixels in an image.

    Equivalent to
    [`PIL.Image.Image.effect_spread()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.effect_spread)

    Args:
        distance: The distance to spread pixels.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.transpose, is_method=True)
def transpose(self: PIL.Image.Image, method: int) -> PIL.Image.Image:
    """
    Transpose the image.

    Equivalent to
    [`PIL.Image.Image.transpose()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose)

    Args:
        method: The transpose method. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose)
            for a list of supported methods.
    """
    pass


@rotate.conditional_return_type
@effect_spread.conditional_return_type
@transpose.conditional_return_type
def _(self: Expr) -> ts.ColumnType:
    return self.col_type


@pxt.udf(substitute_fn=PIL.Image.Image.entropy, is_method=True)
def entropy(self: PIL.Image.Image, mask: Optional[PIL.Image.Image] = None, extrema: Optional[list] = None) -> float:
    """
    Returns the entropy of the image, optionally using a mask and extrema.

    Equivalent to
    [`PIL.Image.Image.entropy()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.entropy)

    Args:
        mask: An optional mask image.
        extrema: An optional list of extrema.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.getbands, is_method=True)
def getbands(self: PIL.Image.Image) -> tuple[str]:
    """
    Return a tuple containing the names of the image bands.

    Equivalent to
    [`PIL.Image.Image.getbands()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getbands)
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.getbbox, is_method=True)
def getbbox(self: PIL.Image.Image, *, alpha_only: bool = True) -> tuple[int, int, int, int]:
    """
    Return a bounding box for the non-zero regions of the image.

    Equivalent to [`PIL.Image.Image.getbbox()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getbbox)

    Args:
        alpha_only: If `True`, and the image has an alpha channel, trim transparent pixels. Otherwise,
            trim pixels when all channels are zero.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.getcolors, is_method=True)
def getcolors(self: PIL.Image.Image, maxcolors: int = 256) -> tuple[tuple[int, int, int], int]:
    """
    Return a list of colors used in the image, up to a maximum of `maxcolors`.

    Equivalent to
    [`PIL.Image.Image.getcolors()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getcolors)

    Args:
        maxcolors: The maximum number of colors to return.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.getextrema, is_method=True)
def getextrema(self: PIL.Image.Image) -> tuple[int, int]:
    """
    Return a 2-tuple containing the minimum and maximum pixel values of the image.

    Equivalent to
    [`PIL.Image.Image.getextrema()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getextrema)
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.getpalette, is_method=True)
def getpalette(self: PIL.Image.Image, mode: Optional[str] = None) -> tuple[int]:
    """
    Return the palette of the image, optionally converting it to a different mode.

    Equivalent to
    [`PIL.Image.Image.getpalette()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getpalette)

    Args:
        mode: The mode to convert the palette to.
    """
    pass


@pxt.udf(is_method=True)
def getpixel(self: PIL.Image.Image, xy: list) -> tuple[int]:
    """
    Return the pixel value at the given position. The position `xy` is a tuple containing the x and y coordinates.

    Equivalent to
    [`PIL.Image.Image.getpixel()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getpixel)

    Args:
        xy: The coordinates, given as (x, y).
    """
    # `xy` will be a list; `tuple(xy)` is necessary for pillow 9 compatibility
    return self.getpixel(tuple(xy))


@pxt.udf(substitute_fn=PIL.Image.Image.getprojection, is_method=True)
def getprojection(self: PIL.Image.Image) -> tuple[int]:
    """
    Return two sequences representing the horizontal and vertical projection of the image.

    Equivalent to
    [`PIL.Image.Image.getprojection()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getprojection)
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.histogram, is_method=True)
def histogram(
    self: PIL.Image.Image, mask: Optional[PIL.Image.Image] = None, extrema: Optional[list] = None
) -> list[int]:
    """
    Return a histogram for the image.

    Equivalent to
    [`PIL.Image.Image.histogram()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.histogram)

    Args:
        mask: An optional mask image.
        extrema: An optional list of extrema.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.quantize, is_method=True)
def quantize(
    self: PIL.Image.Image,
    colors: int = 256,
    method: Optional[int] = None,
    kmeans: int = 0,
    palette: Optional[int] = None,
    dither: int = PIL.Image.Dither.FLOYDSTEINBERG,
) -> PIL.Image.Image:
    """
     Convert the image to 'P' mode with the specified number of colors.

     Equivalent to
     [`PIL.Image.Image.quantize()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize)

    Args:
        colors: The number of colors to quantize to.
        method: The quantization method. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize)
            for a list of supported methods.
        kmeans: The number of k-means clusters to use.
        palette: The palette to use.
        dither: The dithering method. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize)
            for a list of supported methods.
    """
    pass


@pxt.udf(substitute_fn=PIL.Image.Image.reduce, is_method=True)
def reduce(self: PIL.Image.Image, factor: int, box: Optional[tuple[int, int, int, int]] = None) -> PIL.Image.Image:
    """
    Reduce the image by the given factor.

    Equivalent to
    [`PIL.Image.Image.reduce()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.reduce)

    Args:
        factor: The reduction factor.
        box: An optional 4-tuple of ints providing the source image region to be reduced. The values must be within
            (0, 0, width, height) rectangle. If omitted or None, the entire source is used.
    """
    pass


@pxt.udf(is_property=True)
def width(self: PIL.Image.Image) -> int:
    return self.width


@pxt.udf(is_property=True)
def height(self: PIL.Image.Image) -> int:
    return self.height


@pxt.udf(is_property=True)
def mode(self: PIL.Image.Image) -> str:
    return self.mode


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

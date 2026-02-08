"""
Pixeltable UDFs for `ImageType`.

Example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.img_col.convert('L')).collect()
```
"""

from typing import Any, Literal

import PIL.Image

import pixeltable as pxt
import pixeltable.type_system as ts
from pixeltable.exprs import Expr
from pixeltable.utils.code import local_public_names
from pixeltable.utils.image import to_base64


@pxt.udf(is_method=True)
def b64_encode(img: PIL.Image.Image, image_format: str = 'png') -> str:
    """
    Convert image to a base64-encoded string.

    Args:
        img: image
        image_format: image format [supported by PIL](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats)
    """
    return to_base64(img, format=image_format)


@pxt.udf(is_method=True)
def alpha_composite(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    """
    Alpha composite `im2` over `im1`.

    Equivalent to [`PIL.Image.alpha_composite()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.alpha_composite)
    """
    return PIL.Image.alpha_composite(im1, im2)


@pxt.udf(is_method=True)
def blend(im1: PIL.Image.Image, im2: PIL.Image.Image, alpha: float) -> PIL.Image.Image:
    """
    Return a new image by interpolating between two input images, using a constant alpha.

    Equivalent to [`PIL.Image.blend()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.blend)
    """
    return PIL.Image.blend(im1, im2, alpha)


@pxt.udf(is_method=True)
def composite(image1: PIL.Image.Image, image2: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
    """
    Return a composite image by blending two images using a mask.

    Equivalent to [`PIL.Image.composite()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.composite)
    """
    return PIL.Image.composite(image1, image2, mask)


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
@pxt.udf(is_method=True)
def crop(
    self: PIL.Image.Image,
    box: tuple[int, int, int, int],
    *,
    margin_factor: float = 1.0,
    padding: int = 0,
    aspect_ratio: str | None = None,
) -> PIL.Image.Image:
    """
    Return a rectangular region from the image.

    When called with just a ``box``, behaves identically to
    [`PIL.Image.Image.crop()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop).

    The optional parameters let you expand the box and/or fit it to a target aspect ratio
    before cropping — all in a single call:

    - ``margin_factor`` scales the box outward from its centre (e.g. ``1.3`` = 30 % larger).
    - ``padding`` adds that many pixels on every side.
    - ``aspect_ratio`` expands the box to match the given ratio (e.g. ``'9:16'``), centred on the
      original box and clamped to image bounds.

    When multiple parameters are provided they are applied in order: scale → pad → fit to aspect.

    Args:
        box: Bounding box as ``(left, upper, right, lower)``.
        margin_factor: Scale factor around the box centre. ``1.0`` = unchanged.
        padding: Pixels to add on every side after scaling.
        aspect_ratio: Target aspect ratio as ``'W:H'`` (e.g. ``'9:16'``, ``'1:1'``).

    Examples:
        Plain crop (same as PIL):

        ```python
        t.select(t.image.crop((10, 10, 200, 200))).collect()
        ```

        Crop with 30 % margin around a detection:

        ```python
        t.add_computed_column(cropped=t.image.crop(t.bbox, margin_factor=1.3))
        ```

        Crop to 9:16 around a detection with padding:

        ```python
        t.add_computed_column(
            cropped=t.image.crop(t.bbox, padding=20, aspect_ratio='9:16')
        )
        ```
    """
    img_w, img_h = self.size

    x1, y1, x2, y2 = box

    # --- expand ---
    if margin_factor != 1.0 or padding != 0:
        if margin_factor != 1.0:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            half_w = (x2 - x1) * margin_factor / 2.0
            half_h = (y2 - y1) * margin_factor / 2.0
            x1 = round(cx - half_w)
            y1 = round(cy - half_h)
            x2 = round(cx + half_w)
            y2 = round(cy + half_h)
        if padding != 0:
            x1 -= padding
            y1 -= padding
            x2 += padding
            y2 += padding
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

    # --- fit to aspect ratio ---
    if aspect_ratio is not None:
        if ':' in aspect_ratio:
            parts = aspect_ratio.split(':')
        elif 'x' in aspect_ratio.lower():
            parts = aspect_ratio.lower().split('x')
        else:
            raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'. Use 'W:H'.")
        ar_w, ar_h = float(parts[0]), float(parts[1])
        target = ar_w / ar_h

        roi_w = float(x2 - x1)
        roi_h = float(y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if (roi_w / max(roi_h, 1)) < target:
            new_w = roi_h * target
            new_h = roi_h
        else:
            new_w = roi_w
            new_h = roi_w / target

        if new_w > img_w:
            new_w = float(img_w)
            new_h = new_w / target
        if new_h > img_h:
            new_h = float(img_h)
            new_w = new_h * target

        x1 = cx - new_w / 2
        y1 = cy - new_h / 2
        x2 = cx + new_w / 2
        y2 = cy + new_h / 2

        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        elif x2 > img_w:
            x1 -= x2 - img_w
            x2 = float(img_w)
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        elif y2 > img_h:
            y1 -= y2 - img_h
            y2 = float(img_h)

        x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    return self.crop((x1, y1, x2, y2))


@crop.conditional_return_type
def _(
    self: Expr,
    box: tuple[int, int, int, int],
    margin_factor: float = 1.0,
    padding: int = 0,
    aspect_ratio: str | None = None,
) -> ts.ColumnType:
    input_type = self.col_type
    assert isinstance(input_type, ts.ImageType)
    # Can only compute static size for plain crop (no expand/aspect)
    if (
        margin_factor == 1.0
        and padding == 0
        and aspect_ratio is None
        and isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(x, int) for x in box)
    ):
        return ts.ImageType(size=(box[2] - box[0], box[3] - box[1]), mode=input_type.mode, nullable=input_type.nullable)
    return ts.ImageType(mode=input_type.mode, nullable=input_type.nullable)


# Image.getchannel()
@pxt.udf(is_method=True)
def getchannel(self: PIL.Image.Image, channel: int) -> PIL.Image.Image:
    """
    Return an L-mode image containing a single channel of the original image.

    Equivalent to
    [`PIL.Image.Image.getchannel()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getchannel)

    Args:
        channel: The channel to extract. This is a 0-based index.
    """
    return self.getchannel(channel)


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
def point(self: PIL.Image.Image, lut: list[int], mode: str | None = None) -> PIL.Image.Image:
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


@pxt.udf(is_method=True)
def effect_spread(self: PIL.Image.Image, distance: int) -> PIL.Image.Image:
    """
    Randomly spread pixels in an image.

    Equivalent to
    [`PIL.Image.Image.effect_spread()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.effect_spread)

    Args:
        distance: The distance to spread pixels.
    """
    return self.effect_spread(distance)


@pxt.udf(is_method=True)
def transpose(self: PIL.Image.Image, method: Literal[0, 1, 2, 3, 4, 5, 6]) -> PIL.Image.Image:
    """
    Transpose the image.

    Equivalent to
    [`PIL.Image.Image.transpose()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose)

    Args:
        method: The transpose method. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transpose)
            for a list of supported methods.
    """
    return self.transpose(method)


@rotate.conditional_return_type
@effect_spread.conditional_return_type
@transpose.conditional_return_type
def _(self: Expr) -> ts.ColumnType:
    return self.col_type


@pxt.udf(is_method=True)
def entropy(self: PIL.Image.Image, mask: PIL.Image.Image | None = None, extrema: list | None = None) -> float:
    """
    Returns the entropy of the image, optionally using a mask and extrema.

    Equivalent to
    [`PIL.Image.Image.entropy()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.entropy)

    Args:
        mask: An optional mask image.
        extrema: An optional list of extrema.
    """
    return self.entropy(mask, extrema)  # type: ignore[arg-type]


@pxt.udf(is_method=True)
def getbands(self: PIL.Image.Image) -> tuple[str, ...]:
    """
    Return a tuple containing the names of the image bands.

    Equivalent to
    [`PIL.Image.Image.getbands()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getbands)
    """
    return self.getbands()


@pxt.udf(is_method=True)
def getbbox(self: PIL.Image.Image, *, alpha_only: bool = True) -> tuple[int, int, int, int] | None:
    """
    Return a bounding box for the non-zero regions of the image.

    Equivalent to [`PIL.Image.Image.getbbox()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getbbox)

    Args:
        alpha_only: If `True`, and the image has an alpha channel, trim transparent pixels. Otherwise,
            trim pixels when all channels are zero.
    """
    return self.getbbox(alpha_only=alpha_only)


@pxt.udf(is_method=True)
def getcolors(self: PIL.Image.Image, maxcolors: int = 256) -> list[tuple[int, int]]:
    """
    Return a list of colors used in the image, up to a maximum of `maxcolors`.

    Equivalent to
    [`PIL.Image.Image.getcolors()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getcolors)

    Args:
        maxcolors: The maximum number of colors to return.
    """
    return self.getcolors(maxcolors)


@pxt.udf(is_method=True)
def getextrema(self: PIL.Image.Image) -> tuple[int, int]:
    """
    Return a 2-tuple containing the minimum and maximum pixel values of the image.

    Equivalent to
    [`PIL.Image.Image.getextrema()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getextrema)
    """
    return self.getextrema()


@pxt.udf(is_method=True)
def getpalette(self: PIL.Image.Image, mode: str | None = None) -> list[int] | None:
    """
    Return the palette of the image, optionally converting it to a different mode.

    Equivalent to
    [`PIL.Image.Image.getpalette()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getpalette)

    Args:
        mode: The mode to convert the palette to.
    """
    return self.getpalette(mode)


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


@pxt.udf(is_method=True)
def getprojection(self: PIL.Image.Image) -> tuple[list[int], list[int]]:
    """
    Return two sequences representing the horizontal and vertical projection of the image.

    Equivalent to
    [`PIL.Image.Image.getprojection()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.getprojection)
    """
    return self.getprojection()


@pxt.udf(is_method=True)
def histogram(self: PIL.Image.Image, mask: PIL.Image.Image | None = None, extrema: list | None = None) -> list[int]:
    """
    Return a histogram for the image.

    Equivalent to
    [`PIL.Image.Image.histogram()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.histogram)

    Args:
        mask: An optional mask image.
        extrema: An optional list of extrema.
    """
    return self.histogram(mask, extrema)  # type: ignore[arg-type]


@pxt.udf(is_method=True)
def quantize(
    self: PIL.Image.Image,
    colors: int = 256,
    method: Literal[0, 1, 2, 3] | None = None,
    kmeans: int = 0,
    palette: PIL.Image.Image | None = None,
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
    return self.quantize(colors, method, kmeans, palette, dither)


@pxt.udf(is_method=True)
def reduce(self: PIL.Image.Image, factor: int, box: tuple[int, int, int, int] | None = None) -> PIL.Image.Image:
    """
    Reduce the image by the given factor.

    Equivalent to
    [`PIL.Image.Image.reduce()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.reduce)

    Args:
        factor: The reduction factor.
        box: An optional 4-tuple of ints providing the source image region to be reduced. The values must be within
            (0, 0, width, height) rectangle. If omitted or None, the entire source is used.
    """
    return self.reduce(factor, box)


@pxt.udf(is_method=True)
def thumbnail(
    self: PIL.Image.Image,
    size: tuple[int, int],
    resample: int = PIL.Image.Resampling.LANCZOS,
    reducing_gap: float | None = 2.0,
) -> PIL.Image.Image:
    """
    Create a thumbnail of the image.

    Equivalent to
    [`PIL.Image.Image.thumbnail()`](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail)

    Args:
        size: The size of the thumbnail, as a tuple of (width, height).
        resample: The resampling filter to use. See the
            [Pillow documentation](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.thumbnail)
            for a list of supported filters.
        reducing_gap: The reducing gap to use.
    """
    result = self.copy()
    result.thumbnail(size, PIL.Image.Resampling(resample), reducing_gap)
    return result


@pxt.udf(is_property=True)
def width(self: PIL.Image.Image) -> int:
    """
    Return the width of the image.
    """
    return self.width


@pxt.udf(is_property=True)
def height(self: PIL.Image.Image) -> int:
    """
    Return the height of the image.
    """
    return self.height


@pxt.udf(is_property=True)
def mode(self: PIL.Image.Image) -> str:
    """
    Return the image mode.
    """
    return self.mode


# =============================================================================
# Bounding Box Utilities
# =============================================================================


@pxt.udf
def expand_bbox(
    bbox: tuple[int, int, int, int], img_width: int, img_height: int, *, padding: int = 0, margin_factor: float = 1.0
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box by a scale factor and/or pixel padding, clamped to image bounds.

    The bounding box uses PIL convention: ``(left, upper, right, lower)``. When ``margin_factor`` is greater
    than 1.0 the box is scaled outward from its centre. When ``padding`` is non-zero that many pixels are
    added on every side. Both may be combined: the scale factor is applied first, then padding is added.

    Args:
        bbox: Bounding box as ``(left, upper, right, lower)``.
        img_width: Width of the source image in pixels.
        img_height: Height of the source image in pixels.
        padding: Pixels to add on every side of the box after scaling.
        margin_factor: Scale factor applied around the box centre. ``1.0`` = unchanged, ``1.3`` = 30% larger.

    Returns:
        Expanded bounding box as ``(left, upper, right, lower)``, clamped to image bounds.

    Examples:
        Expand a detection box by 30% before cropping:

        ```python
        t.add_computed_column(
            padded=expand_bbox(t.bbox, t.image.width, t.image.height, margin_factor=1.3)
        )
        t.add_computed_column(cropped=t.image.crop(t.padded))
        ```

        Add 20 pixels of uniform padding:

        ```python
        t.add_computed_column(
            padded=expand_bbox(t.bbox, t.image.width, t.image.height, padding=20)
        )
        ```
    """
    x1, y1, x2, y2 = bbox

    if margin_factor != 1.0:
        if margin_factor <= 0:
            raise ValueError(f'margin_factor must be positive, got {margin_factor}')
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_w = (x2 - x1) * margin_factor / 2.0
        half_h = (y2 - y1) * margin_factor / 2.0
        x1 = round(cx - half_w)
        y1 = round(cy - half_h)
        x2 = round(cx + half_w)
        y2 = round(cy + half_h)

    if padding != 0:
        x1 -= padding
        y1 -= padding
        x2 += padding
        y2 += padding

    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))

    return (x1, y1, x2, y2)


@pxt.udf
def rescale_bbox(
    bbox: tuple[int, int, int, int], from_size: tuple[int, int], to_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """
    Rescale bounding box coordinates proportionally after an image resize.

    Use this to keep detection data aligned with an image after calling ``resize()``.

    Args:
        bbox: Bounding box as ``(left, upper, right, lower)``.
        from_size: Original image dimensions as ``(width, height)``.
        to_size: New image dimensions as ``(width, height)``.

    Returns:
        Rescaled bounding box as ``(left, upper, right, lower)``.

    Examples:
        Keep a detection box aligned after resizing:

        ```python
        t.add_computed_column(resized=t.image.resize((640, 480)))
        t.add_computed_column(
            resized_bbox=rescale_bbox(t.bbox, [t.image.width, t.image.height], (640, 480))
        )
        ```
    """
    x1, y1, x2, y2 = bbox
    orig_w, orig_h = from_size
    new_w, new_h = to_size

    if orig_w == 0 or orig_h == 0:
        raise ValueError('Original image dimensions must be non-zero.')

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    return (round(x1 * scale_x), round(y1 * scale_y), round(x2 * scale_x), round(y2 * scale_y))


@pxt.udf
def offset_bbox(
    bbox: tuple[int, int, int, int], crop_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """
    Translate bounding box coordinates into a cropped image's coordinate space.

    After cropping an image, detection bounding boxes still reference the *original* pixel
    coordinates. This function offsets and clips the box so it aligns with the cropped image.
    Returns ``None`` if the bounding box falls entirely outside the crop region.

    Args:
        bbox: Bounding box as ``(left, upper, right, lower)`` in original image coordinates.
        crop_box: Crop region as ``(left, upper, right, lower)`` that was passed to ``crop()``.

    Returns:
        Bounding box in the cropped image's coordinate space, or ``None`` if fully outside.

    Examples:
        Keep a detection box aligned after cropping:

        ```python
        t.add_computed_column(cropped=t.image.crop(t.crop_region))
        t.add_computed_column(cropped_bbox=offset_bbox(t.bbox, t.crop_region))
        ```
    """
    bx1, by1, bx2, by2 = bbox
    cx1, cy1, cx2, cy2 = crop_box

    nx1 = bx1 - cx1
    ny1 = by1 - cy1
    nx2 = bx2 - cx1
    ny2 = by2 - cy1

    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    nx1 = max(0, min(nx1, crop_w))
    ny1 = max(0, min(ny1, crop_h))
    nx2 = max(0, min(nx2, crop_w))
    ny2 = max(0, min(ny2, crop_h))

    if nx1 >= nx2 or ny1 >= ny2:
        return None

    return (nx1, ny1, nx2, ny2)


@pxt.udf
def fit_bbox_to_aspect(
    bbox: tuple[int, int, int, int], frame_width: int, frame_height: int, *, aspect_ratio: str
) -> tuple[int, int, int, int]:
    """
    Compute a crop region that contains a bounding box and matches a target aspect ratio.

    The returned region is centred on the bounding box, expanded to satisfy the requested
    aspect ratio, and clamped to the frame bounds. This is the building block for
    aspect-ratio-aware cropping of images and videos: compute the box here, then pass it
    to ``image.crop()`` or ``video.crop()``.

    Args:
        bbox: Subject bounding box as ``(left, upper, right, lower)``.
        frame_width: Width of the source frame in pixels.
        frame_height: Height of the source frame in pixels.
        aspect_ratio: Target aspect ratio as a string, e.g. ``'9:16'``, ``'16:9'``, ``'1:1'``.

    Returns:
        Crop region as ``(left, upper, right, lower)`` matching the target aspect ratio.

    Examples:
        Compute a 9:16 crop around a detected subject, then crop the image:

        ```python
        t.add_computed_column(
            crop_9x16=fit_bbox_to_aspect(
                t.bbox, t.image.width, t.image.height, aspect_ratio='9:16'
            )
        )
        t.add_computed_column(cropped=t.image.crop(t.crop_9x16))
        ```

        Generate multiple aspect ratios from the same detection:

        ```python
        t.add_computed_column(
            crop_16x9=fit_bbox_to_aspect(
                t.bbox, t.image.width, t.image.height, aspect_ratio='16:9'
            )
        )
        t.add_computed_column(
            crop_1x1=fit_bbox_to_aspect(
                t.bbox, t.image.width, t.image.height, aspect_ratio='1:1'
            )
        )
        ```
    """
    if ':' in aspect_ratio:
        parts = aspect_ratio.split(':')
    elif 'x' in aspect_ratio.lower():
        parts = aspect_ratio.lower().split('x')
    else:
        raise ValueError(f"Invalid aspect_ratio format '{aspect_ratio}'. Use 'W:H' (e.g. '9:16').")

    if len(parts) != 2:
        raise ValueError(f"Invalid aspect_ratio format '{aspect_ratio}'. Expected two numbers separated by ':'.")

    try:
        ar_w = float(parts[0])
        ar_h = float(parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric values in aspect_ratio '{aspect_ratio}'.") from e

    if ar_h == 0:
        raise ValueError('Height component of aspect_ratio cannot be zero.')

    target = ar_w / ar_h

    x1, y1, x2, y2 = bbox
    roi_w = float(x2 - x1)
    roi_h = float(y2 - y1)
    if roi_w < 1 or roi_h < 1:
        raise ValueError('Bounding box must have non-zero dimensions.')

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Expand the smaller dimension to match the target aspect ratio
    if (roi_w / roi_h) < target:
        new_w = roi_h * target
        new_h = roi_h
    else:
        new_w = roi_w
        new_h = roi_w / target

    # Constrain to frame bounds
    if new_w > frame_width:
        new_w = float(frame_width)
        new_h = new_w / target
    if new_h > frame_height:
        new_h = float(frame_height)
        new_w = new_h * target

    # Centre the crop region on the subject
    c_x1 = cx - new_w / 2
    c_y1 = cy - new_h / 2
    c_x2 = cx + new_w / 2
    c_y2 = cy + new_h / 2

    # Shift to stay within frame
    if c_x1 < 0:
        c_x2 -= c_x1
        c_x1 = 0.0
    elif c_x2 > frame_width:
        c_x1 -= c_x2 - frame_width
        c_x2 = float(frame_width)

    if c_y1 < 0:
        c_y2 -= c_y1
        c_y1 = 0.0
    elif c_y2 > frame_height:
        c_y1 -= c_y2 - frame_height
        c_y2 = float(frame_height)

    return (round(c_x1), round(c_y1), round(c_x2), round(c_y2))


def tile_iterator(
    image: Any, tile_size: tuple[int, int], *, overlap: tuple[int, int] = (0, 0)
) -> tuple[type[pxt.iterators.ComponentIterator], dict[str, Any]]:
    """
    Iterator over tiles of an image. Each image will be divided into tiles of size `tile_size`, and the tiles will be
    iterated over in row-major order (left-to-right, then top-to-bottom). An optional `overlap` parameter may be
    specified. If the tiles do not exactly cover the image, then the rightmost and bottommost tiles will be padded with
    blackspace, so that the output images all have the exact size `tile_size`.

    Args:
        image: Image to split into tiles.
        tile_size: Size of each tile, as a pair of integers `[width, height]`.
        overlap: Amount of overlap between adjacent tiles, as a pair of integers `[width, height]`.

    Examples:
        This example assumes an existing table `tbl` with a column `img` of type `pxt.Image`.

        Create a view that splits all images into 256x256 tiles with 32 pixels of overlap:

        >>> pxt.create_view(
        ...     'image_tiles',
        ...     tbl,
        ...     iterator=image_tile_iterator(tbl.img, tile_size=(256, 256), overlap=(32, 32))
        ... )
    """
    kwargs: dict[str, Any] = {}
    if overlap != (0, 0):
        kwargs['overlap'] = overlap
    return pxt.iterators.image.TileIterator._create(image=image, tile_size=tile_size, **kwargs)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

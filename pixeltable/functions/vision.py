"""
Pixeltable UDFs for Computer Vision.

Example:
```python
import pixeltable as pxt
from pixeltable.functions import vision as pxtv

t = pxt.get_table(...)
t.select(
    pxtv.bboxes_draw(t.img, boxes=t.boxes, labels=t.labels)
).collect()
```
"""

import colorsys
import hashlib
import itertools
import re
from collections import defaultdict
from typing import Any, Literal

import numpy as np
import PIL.Image
import PIL.ImageColor
import PIL.ImageDraw
import PIL.ImageFont

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


# the following function has been adapted from MMEval
# (sources at https://github.com/open-mmlab/mmeval)
# Copyright (c) OpenMMLab. All rights reserved.
def __calculate_bboxes_area(bboxes: np.ndarray) -> np.ndarray:
    """Calculate area of bounding boxes.

    Args:
        bboxes (numpy.ndarray): The bboxes with shape (n, 4) or (4, ) in 'xyxy' format.
     Returns:
        numpy.ndarray: The area of bboxes.
    """
    bboxes_w = bboxes[..., 2] - bboxes[..., 0]
    bboxes_h = bboxes[..., 3] - bboxes[..., 1]
    areas = bboxes_w * bboxes_h
    return areas


# the following function has been adapted from MMEval
# (sources at https://github.com/open-mmlab/mmeval)
# Copyright (c) OpenMMLab. All rights reserved.
def __calculate_overlaps(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
    """Calculate the overlap between each bbox of bboxes1 and bboxes2.

    Args:
        bboxes1 (numpy.ndarray): The bboxes with shape (n, 4) in 'xyxy' format.
        bboxes2 (numpy.ndarray): The bboxes with shape (k, 4) in 'xyxy' format.
    Returns:
        numpy.ndarray: IoUs or IoFs with shape (n, k).
    """
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    overlaps = np.zeros((rows, cols), dtype=np.float32)

    if rows * cols == 0:
        return overlaps

    if bboxes1.shape[0] > bboxes2.shape[0]:
        # Swap bboxes for faster calculation.
        bboxes1, bboxes2 = bboxes2, bboxes1
        overlaps = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    else:
        exchange = False

    # Calculate the bboxes area.
    area1 = __calculate_bboxes_area(bboxes1)
    area2 = __calculate_bboxes_area(bboxes2)
    eps = np.finfo(np.float32).eps

    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap_w = np.maximum(x_end - x_start, 0)
        overlap_h = np.maximum(y_end - y_start, 0)
        overlap = overlap_w * overlap_h

        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        overlaps[i, :] = overlap / union
    return overlaps if not exchange else overlaps.T


# the following function has been adapted from MMEval
# (sources at https://github.com/open-mmlab/mmeval)
# Copyright (c) OpenMMLab. All rights reserved.
def __calculate_image_tpfp(
    pred_bboxes: np.ndarray, pred_scores: np.ndarray, gt_bboxes: np.ndarray, min_iou: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the true positive and false positive on an image.

    Args:
        pred_bboxes (numpy.ndarray): Predicted bboxes of this image, with
            shape (N, 5). The scores The predicted score of the bbox is
            concatenated behind the predicted bbox.
        gt_bboxes (numpy.ndarray): Ground truth bboxes of this image, with
            shape (M, 4).
        min_iou (float): The IoU threshold.

    Returns:
        tuple (tp, fp):
        tp: Shape (N,), the true positive flag of each predicted bbox on this image.
        fp: Shape (N,), the false positive flag of each predicted bbox on this image.
    """
    # Step 1. Concatenate `gt_bboxes` and `ignore_gt_bboxes`, then set
    # the `ignore_gt_flags`.
    # all_gt_bboxes = np.concatenate((gt_bboxes, ignore_gt_bboxes))
    # ignore_gt_flags = np.concatenate((np.zeros(
    #     (gt_bboxes.shape[0], 1),
    #     dtype=bool), np.ones((ignore_gt_bboxes.shape[0], 1), dtype=bool)))

    # Step 2. Initialize the `tp` and `fp` arrays.
    num_preds = pred_bboxes.shape[0]
    tp = np.zeros(num_preds, dtype=np.int8)
    fp = np.zeros(num_preds, dtype=np.int8)

    # Step 3. If there are no gt bboxes in this image, then all pred bboxes
    # within area range are false positives.
    if gt_bboxes.shape[0] == 0:
        fp[...] = 1
        return tp, fp

    # Step 4. Calculate the IoUs between the predicted bboxes and the
    # ground truth bboxes.
    ious = __calculate_overlaps(pred_bboxes, gt_bboxes)
    # For each pred bbox, the max iou with all gts.
    ious_max = ious.max(axis=1)
    # For each pred bbox, which gt overlaps most with it.
    ious_argmax = ious.argmax(axis=1)
    # Sort all pred bbox in descending order by scores.
    sorted_indices = np.argsort(-pred_scores)

    # Step 5. Count the `tp` and `fp` of each iou threshold and area range.
    # The flags that gt bboxes have been matched.
    gt_covered_flags = np.zeros(gt_bboxes.shape[0], dtype=bool)

    # Count the prediction bboxes in order of decreasing score.
    for pred_bbox_idx in sorted_indices:
        if ious_max[pred_bbox_idx] >= min_iou:
            matched_gt_idx = ious_argmax[pred_bbox_idx]
            if not gt_covered_flags[matched_gt_idx]:
                tp[pred_bbox_idx] = 1
                gt_covered_flags[matched_gt_idx] = True
            else:
                # This gt bbox has been matched and counted as fp.
                fp[pred_bbox_idx] = 1
        else:
            fp[pred_bbox_idx] = 1

    return tp, fp


@pxt.udf
def eval_detections(
    pred_bboxes: list[list[int]],
    pred_labels: list[int],
    pred_scores: list[float],
    gt_bboxes: list[list[int]],
    gt_labels: list[int],
    min_iou: float = 0.5,
) -> list[dict]:
    """
    Evaluates the performance of a set of predicted bounding boxes against a set of ground truth bounding boxes.

    Args:
        pred_bboxes: List of predicted bounding boxes, each represented as [xmin, ymin, xmax, ymax].
        pred_labels: List of predicted labels.
        pred_scores: List of predicted scores.
        gt_bboxes: List of ground truth bounding boxes, each represented as [xmin, ymin, xmax, ymax].
        gt_labels: List of ground truth labels.
        min_iou: Minimum intersection-over-union (IoU) threshold for a predicted bounding box to be
            considered a true positive.

    Returns:
        A list of dictionaries, one per label class, with the following structure:
        ```python
        {
            'min_iou': float,  # The value of `min_iou` used for the detections
            'class': int,  # The label class
            # List of 1's and 0's indicating true positives for each
            # predicted bounding box of this class
            'tp': list[int],
            # List of 1's and 0's indicating false positives for each
            # predicted bounding box of this class; `fp[n] == 1 - tp[n]`
            'fp': list[int],
            # List of predicted scores for each bounding box of this class
            'scores': list[float],
            'num_gts': int,  # Number of ground truth bounding boxes of this class
        }
        ```
    """
    class_idxs = list(set(pred_labels + gt_labels))
    result: list[dict] = []
    pred_bboxes_arr = np.asarray(pred_bboxes)
    pred_classes_arr = np.asarray(pred_labels)
    pred_scores_arr = np.asarray(pred_scores)
    gt_bboxes_arr = np.asarray(gt_bboxes)
    gt_classes_arr = np.asarray(gt_labels)
    for class_idx in class_idxs:
        pred_filter = pred_classes_arr == class_idx
        gt_filter = gt_classes_arr == class_idx
        class_pred_scores = pred_scores_arr[pred_filter]
        tp, fp = __calculate_image_tpfp(
            pred_bboxes_arr[pred_filter], class_pred_scores, gt_bboxes_arr[gt_filter], min_iou
        )
        ordered_class_pred_scores = -np.sort(-class_pred_scores)
        result.append(
            {
                'min_iou': min_iou,
                'class': class_idx,
                'tp': tp.tolist(),
                'fp': fp.tolist(),
                'scores': ordered_class_pred_scores.tolist(),
                'num_gts': gt_filter.sum().item(),
            }
        )
    return result


@pxt.uda
class mean_ap(pxt.Aggregator):
    """
    Calculates the mean average precision (mAP) over
    [`eval_detections()`][pixeltable.functions.vision.eval_detections] results.

    __Parameters:__

    - `eval_dicts` (list[dict]): List of dictionaries as returned by
        [`eval_detections()`][pixeltable.functions.vision.eval_detections].

    __Returns:__

    - A `dict[int, float]` mapping each label class to an average precision (AP) value for that class.
    """

    def __init__(self) -> None:
        self.class_tpfp: dict[int, list[dict]] = defaultdict(list)

    def update(self, eval_dicts: list[dict]) -> None:
        for eval_dict in eval_dicts:
            class_idx = eval_dict['class']
            self.class_tpfp[class_idx].append(eval_dict)

    def value(self) -> dict:
        eps = np.finfo(np.float32).eps
        result: dict[int, float] = {}
        for class_idx, tpfp in self.class_tpfp.items():
            tp = np.concatenate([x['tp'] for x in tpfp], axis=0)
            fp = np.concatenate([x['fp'] for x in tpfp], axis=0)
            num_gts = np.sum([x['num_gts'] for x in tpfp])
            scores = np.concatenate([np.asarray(x['scores']) for x in tpfp])
            sorted_idxs = np.argsort(-scores)
            tp_cumsum = tp[sorted_idxs].cumsum()
            fp_cumsum = fp[sorted_idxs].cumsum()
            precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
            recall = tp_cumsum / np.maximum(num_gts, eps)

            mrec = np.hstack((0, recall, 1))
            mpre = np.hstack((0, precision, 0))
            for i in range(mpre.shape[0] - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            ind = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[ind + 1] - mrec[ind]) * mpre[ind + 1])
            result[class_idx] = ap.item()
        return result


def __create_label_colors(labels: list[Any]) -> dict[Any, str]:
    """
    Create random colors for labels such that a particular label always gets the same color.

    Returns:
        dict mapping labels to colors
    """
    distinct_labels = set(labels)
    result: dict[Any, str] = {}
    for label in distinct_labels:
        # consistent hash for the label
        label_hash = int(hashlib.md5(str(label).encode()).hexdigest(), 16)
        hue = (label_hash % 360) / 360.0
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
        result[label] = hex_color
    return result


@pxt.udf
def bboxes_draw(
    img: PIL.Image.Image,
    boxes: list[list[int]],
    *,
    labels: list[Any] | None = None,
    color: str | None = None,
    box_colors: list[str] | None = None,
    alpha: float | None = None,
    fill: bool = False,
    fill_alpha: float | None = None,
    width: int = 1,
    font: str | None = None,
    font_size: int | None = None,
) -> PIL.Image.Image:
    """
    Draws bounding boxes on the given image.

    Labels can be any type that supports `str()` and is hashable (e.g., strings, ints, etc.).

    Colors can be specified as common HTML color names (e.g., 'red') supported by PIL's
    [`ImageColor`](https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#imagecolor-module) module or as
    RGB/RGBA hex codes (e.g., '#FF0000', '#FF0000FF'). If opacity isn't specified in the color string and
    `alpha`/`fill_alpha` is `None`, defaults to 1.0 for box borders and 0.5 for filled boxes.

    If no colors are specified, this function randomly assigns each label a specific color based on a hash of the label.

    Args:
        img: The image on which to draw the bounding boxes.
        boxes: List of bounding boxes, each represented as [xmin, ymin, xmax, ymax].
        labels: List of labels for each bounding box.
        color: Single color to be used for all bounding boxes and labels.
        box_colors: List of colors, one per bounding box.
        alpha: Opacity (0-1) of the bounding box borders and labels. If non-`None`, overrides any alpha in
            `color`/`box_colors`.
        fill: Whether to fill the bounding boxes with color.
        fill_alpha: Opacity (0-1) of the bounding box fill. If non-`None`, overrides any alpha in
            `color`/`box_colors`.
        width: Width of the bounding box borders.
        font: Name of a system font or path to a TrueType font file, as required by
            [`PIL.ImageFont.truetype()`](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.truetype).
            If `None`, uses the default provided by
            [`PIL.ImageFont.load_default()`](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.load_default).
        font_size: Size of the font used for labels in points. Only used in conjunction with non-`None` `font` argument.

    Returns:
        The image with bounding boxes drawn on it.
    """
    if len(boxes) == 0:
        return img

    color_params = sum([color is not None, box_colors is not None])
    if color_params > 1:
        raise ValueError("Only one of 'color' or 'box_colors' can be set")

    # ensure the number of labels matches the number of boxes
    num_boxes = len(boxes)
    if labels is None:
        labels = [None] * num_boxes
    elif len(labels) != num_boxes:
        raise ValueError('Number of boxes and labels must match')

    if box_colors is not None:
        if len(box_colors) != num_boxes:
            raise ValueError('Number of boxes and box colors must match')
    elif color is not None:
        box_colors = [color] * num_boxes
    else:
        label_colors = __create_label_colors(labels)
        box_colors = [label_colors[label] for label in labels]

    rgb_box_colors = [PIL.ImageColor.getrgb(c) for c in box_colors]
    rgba_border_colors: list[tuple[int, int, int, int]]
    if alpha is not None:
        # override any alpha in rgb_box_colors
        int_alpha = int(alpha * 255)
        rgba_border_colors = [(c[0], c[1], c[2], int_alpha) for c in rgb_box_colors]
    else:
        # default to full opacity if alpha is missing
        rgba_border_colors = [(*c, 255) if len(c) == 3 else c for c in rgb_box_colors]

    rgba_fill_colors: list[tuple[int, int, int, int]] = []
    if fill:
        if fill_alpha is not None:
            int_fill_alpha = int(fill_alpha * 255)
            rgba_fill_colors = [(c[0], c[1], c[2], int_fill_alpha) for c in rgb_box_colors]
        else:
            # default to semi-transparent if alpha is missing
            rgba_fill_colors = [(*c, 127) if len(c) == 3 else c for c in rgb_box_colors]

    # set default font if not provided
    txt_font: PIL.ImageFont.ImageFont | PIL.ImageFont.FreeTypeFont = (
        PIL.ImageFont.load_default() if font is None else PIL.ImageFont.truetype(font=font, size=font_size or 10)
    )

    img_to_draw = img.copy()
    draw = PIL.ImageDraw.Draw(img_to_draw, 'RGBA')

    # Draw bounding boxes
    for i, bbox in enumerate(boxes):
        # determine color for the current box and label
        border_color = rgba_border_colors[i % len(box_colors)]

        if fill:
            fill_color = rgba_fill_colors[i % len(box_colors)]
            draw.rectangle(bbox, outline=border_color, width=width, fill=fill_color)  # type: ignore[arg-type]
        else:
            draw.rectangle(bbox, outline=border_color, width=width)  # type: ignore[arg-type]

    # Now draw labels separately, so they are not obscured by the boxes
    for bbox, label in zip(boxes, labels):
        if label is not None:
            label_str = str(label)
            _, _, text_width, text_height = draw.textbbox((0, 0), label_str, font=txt_font)
            if bbox[1] - text_height - 2 >= 0:
                # draw text above the box
                y = bbox[1] - text_height - 2
            else:
                y = bbox[3]
            if bbox[0] + text_width + 2 < img.width:
                x = bbox[0]
            else:
                x = img.width - text_width - 2
            draw.rectangle((x, y, x + text_width + 1, y + text_height + 1), fill='black')
            draw.text((x, y), label_str, fill='white', font=txt_font)

    return img_to_draw


def _validate_bboxes(bboxes: list, error_prefix: str, validate_range: bool = True) -> bool:
    """Check that bboxes are either all int or all float. Return True for absolute, False for relative."""
    if not all(len(b) == 4 for b in bboxes):
        raise pxt.Error(f'{error_prefix}: each bounding box must have exactly 4 coordinates')
    is_absolute = all(
        isinstance(x, int) and (not validate_range or x >= 0) for x in itertools.chain.from_iterable(bboxes)
    )
    is_relative = all(
        isinstance(x, float) and (not validate_range or (0.0 <= x <= 1.0))
        for x in itertools.chain.from_iterable(bboxes)
    )
    if not (is_absolute or is_relative):
        raise pxt.Error(
            f'{error_prefix}: bounding box coordinates must be either all int'
            f'{" (>= 0)" if validate_range else ""} or all float{" (in [0, 1])" if validate_range else ""}'
        )
    return is_absolute


@pxt.udf
def bboxes_convert(
    bboxes: list,  # should be list[list[int | float]]
    *,
    src_format: Literal['xyxy', 'xywh', 'cxcywh'],
    dst_format: Literal['xyxy', 'xywh', 'cxcywh'],
) -> list:
    """
    Convert a list of bounding boxes from src_format to dst_format.

    Args:
        bboxes: List of bounding boxes, each either specified with absolute pixel coordinates or relative
            coordinates in [0, 1].
        src_format: Source format, one of 'xyxy', 'xywh', 'cxcywh'.
        dst_format: Destination format, one of 'xyxy', 'xywh', 'cxcywh'.

    Returns:
        List of bounding boxes in dst_format.
    """
    if len(bboxes) == 0:
        return []

    if src_format not in ('xyxy', 'xywh', 'cxcywh'):
        raise pxt.Error(f'Invalid src_format: {src_format!r}')
    if dst_format not in ('xyxy', 'xywh', 'cxcywh'):
        raise pxt.Error(f'Invalid dst_format: {dst_format!r}')
    is_absolute = _validate_bboxes(bboxes, 'bboxes_convert()')
    if src_format == dst_format:
        return bboxes

    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    result: np.ndarray
    if src_format == 'xyxy' and dst_format == 'xywh':
        result = np.column_stack([c0, c1, c2 - c0, c3 - c1])
    elif src_format == 'xyxy' and dst_format == 'cxcywh':
        w, h = c2 - c0, c3 - c1
        result = np.column_stack([c0 + w / 2, c1 + h / 2, w, h])
    elif src_format == 'xywh' and dst_format == 'xyxy':
        result = np.column_stack([c0, c1, c0 + c2, c1 + c3])
    elif src_format == 'xywh' and dst_format == 'cxcywh':
        result = np.column_stack([c0 + c2 / 2, c1 + c3 / 2, c2, c3])
    elif src_format == 'cxcywh' and dst_format == 'xyxy':
        result = np.column_stack([c0 - c2 / 2, c1 - c3 / 2, c0 + c2 / 2, c1 + c3 / 2])
    else:  # cxcywh -> xywh
        result = np.column_stack([c0 - c2 / 2, c1 - c3 / 2, c2, c3])

    if is_absolute:
        # don't use round() here, it rounds to the nearest even number
        result = np.floor(result + 0.5).astype(int)
    return result.tolist()


ASPECT_RATIO_RE = re.compile(r'(\d+):(\d+)')


@pxt.udf
def bboxes_resize(
    bboxes: list,  # should be: list[list[int]] | list[list[float]]
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    width: int | None = None,
    height: int | None = None,
    aspect: str | None = None,
    aspect_mode: str | None = None,  # should be Literal['crop', 'pad'] | None
) -> list:
    """
    Resize a list of bounding boxes (center-anchored):

    - to a specified width or height (the other dimension is scaled to maintain the aspect ratio)
    - to a specified aspect ratio

    Only one of `width`, `height`, or `aspect` can be specified.

    Args:
        bboxes: List of bounding boxes, each either specified with absolute pixel coordinates or relative
            coordinates in [0, 1].
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        width: Target width. Pass an `int` for absolute pixels or a `float` for relative coordinates.
        height: Target height. Pass an `int` for absolute pixels or a `float` for relative coordinates.
        aspect: Target aspect ratio. Pass a `str` like '16:9' or a `float` like 1.78.
        aspect: Target aspect ratio as a string 'W:H' (e.g., '16:9') or a `float`. Resizes either the width
            or height to match the specified aspect ratio, maintaining the other dimension. Requires `aspect_mode`.
        aspect_mode: Either 'crop' or 'pad'. Required when `aspect` is specified. If `crop`, reduces the oversized
            dimension to match the aspect ratio. If `pad`, extends the undersized dimension to match the aspect ratio.

    Returns:
        List of resized bounding boxes in the same format as the input.
    """
    if width is not None and width <= 0:
        raise pxt.Error(f'width must be positive, got {width}')
    if height is not None and height <= 0:
        raise pxt.Error(f'height must be positive, got {height}')
    aspect_f: float | None = None
    if aspect is not None:
        match = ASPECT_RATIO_RE.fullmatch(aspect)
        if match is None:
            raise pxt.Error(f'Invalid aspect ratio: {aspect!r}; expected "W:H"')
        w_val, h_val = int(match.group(1)), int(match.group(2))
        if w_val == 0 or h_val == 0:
            raise pxt.Error(f'Invalid aspect ratio: {aspect!r}; width and height must be positive')
        aspect_f = float(w_val) / float(h_val)
    return _bboxes_resize(bboxes, format, width=width, height=height, aspect=aspect_f, aspect_mode=aspect_mode)


@bboxes_resize.overload
def _(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    width: float | None = None,
    height: float | None = None,
    aspect: float | None = None,
    aspect_mode: str | None = None,
) -> list:
    if width is not None and width <= 0:
        raise pxt.Error(f'width must be positive, got {width}')
    if height is not None and height <= 0:
        raise pxt.Error(f'height must be positive, got {height}')
    if aspect is not None and aspect <= 0:
        raise pxt.Error(f'aspect must be positive, got {aspect}')

    return _bboxes_resize(bboxes, format, width_f=width, height_f=height, aspect=aspect, aspect_mode=aspect_mode)


def _bboxes_resize(
    bboxes: list,  # should be: list[list[int]] | list[list[float]]
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    width: int | None = None,
    width_f: float | None = None,
    height: int | None = None,
    height_f: float | None = None,
    aspect: float | None = None,
    aspect_mode: str | None = None,  # should be Literal['crop', 'pad'] | None
) -> list:
    if len(bboxes) == 0:
        return []

    # TODO: this is a lot of repeated per-call validation; find a way to do this at plan generation time, where possible
    assert width is None or width > 0
    assert width_f is None or width_f > 0
    assert height is None or height > 0
    assert height_f is None or height_f > 0
    assert aspect is None or aspect > 0
    if aspect_mode is not None and aspect_mode not in ['crop', 'pad']:
        raise pxt.Error(f'Invalid aspect_mode: {aspect_mode!r}; expected "crop" or "pad"')

    has_width = width is not None or width_f is not None
    has_height = height is not None or height_f is not None
    has_aspect = aspect is not None
    if has_width + has_height + has_aspect != 1:
        raise pxt.Error('Exactly one of width, height, or aspect must be specified')
    if has_aspect and aspect_mode is None:
        raise pxt.Error("aspect_mode ('crop' or 'pad') is required when aspect is specified")
    if not has_aspect and aspect_mode is not None:
        raise pxt.Error('aspect_mode is only valid when aspect is specified')

    is_absolute = _validate_bboxes(bboxes, 'bboxes_resize()')
    if is_absolute and (width_f is not None or height_f is not None):
        raise pxt.Error('bboxes_resize(): width/height require relative coordinates, but bboxes use absolute pixels')
    if not is_absolute and (width is not None or height is not None):
        raise pxt.Error(
            'bboxes_resize(): width/height require absolute pixel coordinates, but bboxes use relative coordinates'
        )
    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Convert to cx, cy, w, h
    w: np.ndarray
    h: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    if format == 'xyxy':
        w, h = c2 - c0, c3 - c1
        cx, cy = c0 + w / 2, c1 + h / 2
    elif format == 'xywh':
        w, h = c2, c3
        cx, cy = c0 + w / 2, c1 + h / 2
    elif format == 'cxcywh':
        cx, cy, w, h = c0, c1, c2, c3
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    valid = (w > 0) & (h > 0)
    orig: np.ndarray | None = None
    if not valid.all():
        # save original array for invalid boxes to pass through unchanged
        orig = arr.copy()
    # Replace invalid dimensions with 1.0 to avoid division by zero
    w = np.where(valid, w, 1.0)
    h = np.where(valid, h, 1.0)

    # Resolve the target width/height
    target_w = width if width is not None else width_f
    target_h = height if height is not None else height_f

    if target_w is not None:
        scale = target_w / w
        w = np.full_like(w, target_w)
        h *= scale
    elif target_h is not None:
        scale = target_h / h
        h = np.full_like(h, target_h)
        w *= scale
    else:
        current_aspect = w / h
        if aspect_mode == 'crop':
            # Reduce the oversized dimension
            too_wide = current_aspect > aspect
            new_w = np.where(too_wide, h * aspect, w)
            new_h = np.where(too_wide, h, w / aspect)
        else:  # pad
            # Extend the undersized dimension
            too_wide = current_aspect > aspect
            new_w = np.where(too_wide, w, h * aspect)
            new_h = np.where(too_wide, w / aspect, h)
        w, h = new_w, new_h

    # Convert back to original format.
    # For absolute coordinates, round w/h first, then derive positions from rounded
    # dimensions so that x2-x1==round(w) (xyxy) and x+w is consistent (xywh).
    if is_absolute:
        # don't use round() here, it rounds to the nearest even number
        w = np.floor(w + 0.5)
        h = np.floor(h + 0.5)

    if format == 'xyxy':
        if is_absolute:
            x1 = np.floor(cx - w / 2 + 0.5)
            y1 = np.floor(cy - h / 2 + 0.5)
            result = np.column_stack([x1, y1, x1 + w, y1 + h]).astype(int)
        else:
            result = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    elif format == 'xywh':
        if is_absolute:
            x1 = np.floor(cx - w / 2 + 0.5)
            y1 = np.floor(cy - h / 2 + 0.5)
            result = np.column_stack([x1, y1, w, h]).astype(int)
        else:
            result = np.column_stack([cx - w / 2, cy - h / 2, w, h])
    else:  # cxcywh
        result = np.column_stack([cx, cy, w, h])
        if is_absolute:
            result = np.floor(result + 0.5).astype(int)

    if not valid.all():
        # leave invalid boxes as-is
        result[~valid] = orig[~valid]
    return result.tolist()


@pxt.udf
def bboxes_scale(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    factor: float | None = None,
    x_factor: float | None = None,
    y_factor: float | None = None,
) -> list:
    """
    Re-scale a list of bounding boxes (center-anchored).

    Args:
        bboxes: List of bounding boxes, each either specified with absolute pixel coordinates or relative
            coordinates in [0, 1].
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        factor: Scale factor to apply to both box dimensions.
        x_factor: Scale factor to apply to the box width.
        y_factor: Scale factor to apply to the box height.

    Returns:
        List of scaled bounding boxes in the same format as the input.
    """
    if len(bboxes) == 0:
        return []

    # Parameter validation
    has_factor = factor is not None
    has_x = x_factor is not None
    has_y = y_factor is not None
    if not has_factor and not has_x and not has_y:
        raise pxt.Error('bboxes_scale(): at least one of factor, x_factor, y_factor must be specified')
    if has_factor and (has_x or has_y):
        raise pxt.Error('bboxes_scale(): factor is mutually exclusive with x_factor/y_factor')
    if has_factor and factor <= 0:
        raise pxt.Error('bboxes_scale(): factor must be positive')
    if has_x and x_factor <= 0:
        raise pxt.Error('bboxes_scale(): x_factor must be positive')
    if has_y and y_factor <= 0:
        raise pxt.Error('bboxes_scale(): y_factor must be positive')

    is_absolute = _validate_bboxes(bboxes, 'bboxes_scale()')
    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Convert to cx, cy, w, h
    w: np.ndarray
    h: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    if format == 'xyxy':
        w, h = c2 - c0, c3 - c1
        cx, cy = c0 + w / 2, c1 + h / 2
    elif format == 'xywh':
        w, h = c2, c3
        cx, cy = c0 + w / 2, c1 + h / 2
    elif format == 'cxcywh':
        cx, cy, w, h = c0, c1, c2, c3
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    valid = (w > 0) & (h > 0)
    orig: np.ndarray | None = None
    if not valid.all():
        orig = arr.copy()
    w = np.where(valid, w, 1.0)
    h = np.where(valid, h, 1.0)

    # scale w/h
    if has_factor:
        w *= factor
        h *= factor
    else:
        if has_x:
            w *= x_factor
        if has_y:
            h *= y_factor

    # Convert back to original format
    if is_absolute:
        w = np.floor(w + 0.5)
        h = np.floor(h + 0.5)

    if format == 'xyxy':
        if is_absolute:
            x1 = np.floor(cx - w / 2 + 0.5)
            y1 = np.floor(cy - h / 2 + 0.5)
            result = np.column_stack([x1, y1, x1 + w, y1 + h]).astype(int)
        else:
            result = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    elif format == 'xywh':
        if is_absolute:
            x1 = np.floor(cx - w / 2 + 0.5)
            y1 = np.floor(cy - h / 2 + 0.5)
            result = np.column_stack([x1, y1, w, h]).astype(int)
        else:
            result = np.column_stack([cx - w / 2, cy - h / 2, w, h])
    else:  # cxcywh
        result = np.column_stack([cx, cy, w, h])
        if is_absolute:
            result = np.floor(result + 0.5).astype(int)

    if not valid.all():
        result[~valid] = orig[~valid]
    return result.tolist()


@pxt.udf
def bboxes_pad(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    top: int | None = None,
    bottom: int | None = None,
    left: int | None = None,
    right: int | None = None,
    x: int | None = None,
    y: int | None = None,
) -> list:
    """
    Pad a list of bounding boxes.

    Args:
        bboxes: List of bounding boxes in absolute pixel coordinates.
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        top: Amount to pad at the top, in absolute pixels.
        bottom: Amount to pad at the bottom, in absolute pixels.
        left: Amount to pad at the left, in absolute pixels.
        right: Amount to pad at the right, in absolute pixels.
        x: Amount to pad at the left and right, in absolute pixels.
        y: Amount to pad at the top and bottom, in absolute pixels.

    Returns:
        List of padded bounding boxes in the same format as the input.
    """
    if len(bboxes) == 0:
        return []

    # Parameter validation
    has_x = x is not None
    has_y = y is not None
    has_left = left is not None
    has_right = right is not None
    has_top = top is not None
    has_bottom = bottom is not None
    if has_x and (has_left or has_right):
        raise pxt.Error('bboxes_pad(): x is mutually exclusive with left/right')
    if has_y and (has_top or has_bottom):
        raise pxt.Error('bboxes_pad(): y is mutually exclusive with top/bottom')
    if not (has_x or has_y or has_left or has_right or has_top or has_bottom):
        raise pxt.Error('bboxes_pad(): at least one padding parameter must be specified')

    # Resolve effective padding
    pad_left = x if has_x else (left if has_left else 0)
    pad_right = x if has_x else (right if has_right else 0)
    pad_top = y if has_y else (top if has_top else 0)
    pad_bottom = y if has_y else (bottom if has_bottom else 0)

    for name, val in [('left', pad_left), ('right', pad_right), ('top', pad_top), ('bottom', pad_bottom)]:
        if val < 0:
            raise pxt.Error(f'bboxes_pad(): {name} padding must be >= 0')

    is_absolute = _validate_bboxes(bboxes, 'bboxes_pad()')
    if not is_absolute:
        raise pxt.Error(
            'bboxes_pad(): padding requires absolute pixel coordinates, but bboxes use relative coordinates'
        )

    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Detect degenerate boxes (format-dependent)
    if format == 'xyxy':
        valid = (c2 > c0) & (c3 > c1)
    elif format in ('xywh', 'cxcywh'):
        valid = (c2 > 0) & (c3 > 0)
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    orig: np.ndarray | None = None
    if not valid.all():
        orig = arr.copy()

    if format == 'xyxy':
        c0 = np.where(valid, c0 - pad_left, c0)
        c1 = np.where(valid, c1 - pad_top, c1)
        c2 = np.where(valid, c2 + pad_right, c2)
        c3 = np.where(valid, c3 + pad_bottom, c3)
    elif format == 'xywh':
        c0 = np.where(valid, c0 - pad_left, c0)
        c1 = np.where(valid, c1 - pad_top, c1)
        c2 = np.where(valid, c2 + pad_left + pad_right, c2)
        c3 = np.where(valid, c3 + pad_top + pad_bottom, c3)
    else:  # cxcywh
        c0 = np.where(valid, c0 + (pad_right - pad_left) / 2, c0)
        c1 = np.where(valid, c1 + (pad_bottom - pad_top) / 2, c1)
        c2 = np.where(valid, c2 + pad_left + pad_right, c2)
        c3 = np.where(valid, c3 + pad_top + pad_bottom, c3)

    result = np.floor(np.column_stack([c0, c1, c2, c3]) + 0.5).astype(int)

    if orig is not None:
        result[~valid] = orig[~valid]
    return result.tolist()


@pxt.udf
def bboxes_clip_to_canvas(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    width: int | None = None,
    height: int | None = None,
    min_visibility: float = 0.0,
    min_area: float = 0.0,
) -> list:
    """
    Clip a list of bounding boxes to a canvas of specified size.

    Args:
        bboxes: List of bounding boxes, each either specified with absolute pixel coordinates (`int`) or relative
            coordinates (`float`).
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        width: Canvas width in absolute pixels. Required for absolute coordinates, must not be specified for relative.
        height: Canvas height in absolute pixels. Required for absolute coordinates, must not be specified for relative.
        min_visibility: Minimum fraction of the bounding box that must be visible after clipping. If the visibility
            is less than this value, returns None.
        min_area: Minimum area of the bounding box after clipping. If the area is less than this value, returns None.

    Returns:
        List of clipped bounding boxes in the same format as the input. Boxes that don't meet the
        min_visibility or min_area thresholds are replaced with None.
    """
    if len(bboxes) == 0:
        return []

    is_absolute = _validate_bboxes(bboxes, 'bboxes_clip_to_canvas()', validate_range=False)

    if is_absolute and (width is None or height is None):
        raise pxt.Error('bboxes_clip_to_canvas(): both width and height must be specified for absolute coordinates')
    if not is_absolute and (width is not None or height is not None):
        raise pxt.Error('bboxes_clip_to_canvas(): width/height must not be specified for relative coordinates')
    if not (0.0 <= min_visibility <= 1.0):
        raise pxt.Error('bboxes_clip_to_canvas(): min_visibility must be between 0.0 and 1.0')
    if min_area < 0.0:
        raise pxt.Error('bboxes_clip_to_canvas(): min_area must be >= 0')

    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # Convert to xyxy for clipping
    if format == 'xyxy':
        x1, y1, x2, y2 = c0, c1, c2, c3
    elif format == 'xywh':
        x1, y1, x2, y2 = c0, c1, c0 + c2, c1 + c3
    elif format == 'cxcywh':
        x1, y1, x2, y2 = c0 - c2 / 2, c1 - c3 / 2, c0 + c2 / 2, c1 + c3 / 2
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    # Detect degenerate boxes (zero or negative area)
    valid = (x2 > x1) & (y2 > y1)

    # Original area (for min_visibility)
    orig_area = np.where(valid, (x2 - x1) * (y2 - y1), 0.0)

    # Clip to canvas bounds
    x_max = float(width) if is_absolute else 1.0
    y_max = float(height) if is_absolute else 1.0

    cx1 = np.clip(x1, 0, x_max)
    cy1 = np.clip(y1, 0, y_max)
    cx2 = np.clip(x2, 0, x_max)
    cy2 = np.clip(y2, 0, y_max)

    # Clipped area
    clipped_area = np.maximum(cx2 - cx1, 0) * np.maximum(cy2 - cy1, 0)

    # Determine which boxes survive filtering
    survive = valid.copy()
    if min_visibility > 0.0:
        with np.errstate(divide='ignore', invalid='ignore'):
            visibility = np.where(orig_area > 0, clipped_area / orig_area, 0.0)
        survive &= visibility >= min_visibility
    if min_area > 0.0:
        survive &= clipped_area >= min_area

    # Convert back to original format
    if format == 'xyxy':
        result_arr = np.column_stack([cx1, cy1, cx2, cy2])
    elif format == 'xywh':
        result_arr = np.column_stack([cx1, cy1, cx2 - cx1, cy2 - cy1])
    else:  # cxcywh
        result_arr = np.column_stack([(cx1 + cx2) / 2, (cy1 + cy2) / 2, cx2 - cx1, cy2 - cy1])

    if is_absolute:
        result_arr = np.floor(result_arr + 0.5).astype(int)

    # Degenerate boxes pass through unchanged
    result_arr[~valid] = arr[~valid]

    # Build result: None for filtered valid boxes, passthrough for degenerate boxes
    result: list = []
    for i in range(len(bboxes)):
        if not valid[i]:
            # Degenerate box: pass through unchanged
            result.append(result_arr[i].tolist())
        elif not survive[i]:
            # Valid box that was filtered out
            result.append(None)
        else:
            result.append(result_arr[i].tolist())
    return result


@pxt.udf
def bboxes_crop_canvas(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    canvas_region: list,
    canvas_region_format: Literal['xyxy', 'xywh', 'cxcywh'],
    canvas_width: int | None = None,
    canvas_height: int | None = None,
) -> list:
    """
    Adjust a list of bounding boxes to account for a canvas crop.

    Args:
        bboxes: List of bounding boxes, each either specified with absolute pixel coordinates or relative coordinates.
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        canvas_width: Canvas width.
        canvas_height: Canvas height.
        canvas_region: Canvas region that was cropped, either specified with absolute pixel coordinates or relative
            coordinates, in the format specified by `canvas_region_format`.
        canvas_region_format: Format of the `canvas_region` coordinates, one of 'xyxy', 'xywh', 'cxcywh'.

    Returns:
        List of adjusted bounding boxes in the same format as the input. They can extend beyond the canvas boundaries.
    """
    if len(bboxes) == 0:
        return []

    is_absolute = _validate_bboxes(bboxes, 'bboxes_crop_canvas()', validate_range=False)

    if is_absolute and (canvas_width is None or canvas_height is None):
        raise pxt.Error(
            'bboxes_crop_canvas(): both canvas_width and canvas_height must be specified for absolute coordinates'
        )
    if not is_absolute and (canvas_width is not None or canvas_height is not None):
        raise pxt.Error(
            'bboxes_crop_canvas(): canvas_width/canvas_height must not be specified for relative coordinates'
        )

    # Validate canvas_region
    if not isinstance(canvas_region, list) or len(canvas_region) != 4:
        raise pxt.Error('bboxes_crop_canvas(): canvas_region must be a list of 4 coordinates')

    # normalize to xyxy
    rc0, rc1, rc2, rc3 = canvas_region
    if canvas_region_format == 'xyxy':
        rx1, ry1, rx2, ry2 = rc0, rc1, rc2, rc3
    elif canvas_region_format == 'xywh':
        rx1, ry1, rx2, ry2 = rc0, rc1, rc0 + rc2, rc1 + rc3
    elif canvas_region_format == 'cxcywh':
        rx1, ry1, rx2, ry2 = rc0 - rc2 / 2, rc1 - rc3 / 2, rc0 + rc2 / 2, rc1 + rc3 / 2
    else:
        raise pxt.Error(f'Invalid canvas_region_format: {canvas_region_format!r}')

    if rx2 <= rx1 or ry2 <= ry1:
        raise pxt.Error('bboxes_crop_canvas(): canvas_region must have positive area')
    if is_absolute:
        if rx1 < 0 or ry1 < 0 or rx2 > canvas_width or ry2 > canvas_height:
            raise pxt.Error('bboxes_crop_canvas(): canvas_region extends beyond canvas bounds')
    elif rx1 < 0.0 or ry1 < 0.0 or rx2 > 1.0 or ry2 > 1.0:
        raise pxt.Error('bboxes_crop_canvas(): canvas_region extends beyond canvas bounds')

    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # normalize to xyxy
    x1: np.ndarray
    x2: np.ndarray
    y1: np.ndarray
    y2: np.ndarray
    if format == 'xyxy':
        x1, y1, x2, y2 = c0, c1, c2, c3
    elif format == 'xywh':
        x1, y1, x2, y2 = c0, c1, c0 + c2, c1 + c3
    elif format == 'cxcywh':
        x1, y1, x2, y2 = c0 - c2 / 2, c1 - c3 / 2, c0 + c2 / 2, c1 + c3 / 2
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    # Detect degenerate boxes
    valid = (x2 > x1) & (y2 > y1)

    # translate so crop region top-left becomes origin
    nx1: np.ndarray
    nx2: np.ndarray
    ny1: np.ndarray
    ny2: np.ndarray
    if is_absolute:
        nx1 = x1 - rx1
        ny1 = y1 - ry1
        nx2 = x2 - rx1
        ny2 = y2 - ry1
    else:
        crop_w = rx2 - rx1
        crop_h = ry2 - ry1
        nx1 = (x1 - rx1) / crop_w
        ny1 = (y1 - ry1) / crop_h
        nx2 = (x2 - rx1) / crop_w
        ny2 = (y2 - ry1) / crop_h

    # Convert back to original format
    result: np.ndarray
    if format == 'xyxy':
        result = np.column_stack([nx1, ny1, nx2, ny2])
    elif format == 'xywh':
        result = np.column_stack([nx1, ny1, nx2 - nx1, ny2 - ny1])
    else:  # cxcywh
        result = np.column_stack([(nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1])

    if is_absolute:
        result = np.floor(result + 0.5).astype(int)

    # Degenerate boxes pass through unchanged
    result[~valid] = arr[~valid]

    return result.tolist()


@pxt.udf
def bboxes_resize_canvas(
    bboxes: list,
    format: Literal['xyxy', 'xywh', 'cxcywh'],
    *,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    new_canvas_width: int | None = None,
    new_canvas_height: int | None = None,
    canvas_scale: float | None = None,
    canvas_scale_x: float | None = None,
    canvas_scale_y: float | None = None,
) -> list:
    """
    Adjust a list of bounding boxes to account for a canvas resize. The resize operation can be expressed

    - as absolute pixel dimensions (requires canvas_width, canvas_height, new_canvas_width, new_canvas_height)
    - as relative dimensions (requires at least one of canvas_scale, canvas_scale_x, canvas_scale_y)

    Args:
        bboxes: List of bounding boxes in absolute pixel coordinates.
        format: Format of the bounding box coordinates, one of 'xyxy', 'xywh', 'cxcywh'.
        canvas_width: Original canvas width in absolute pixels.
        canvas_height: Original canvas height in absolute pixels.
        new_canvas_width: New canvas width in absolute pixels. Requires canvas_width/canvas_height to be specified.
        new_canvas_height: New canvas height in absolute pixels. Requires canvas_width/canvas_height to be specified.
        canvas_scale: Scale factor to apply to both canvas dimensions.
        canvas_scale_x: Scale factor to apply to the canvas width.
        canvas_scale_y: Scale factor to apply to the canvas height.

    Returns:
        List of adjusted bounding boxes in the same format as the input.
    """
    # Early exit
    if len(bboxes) == 0:
        return []

    is_absolute = _validate_bboxes(bboxes, 'bboxes_resize_canvas()', validate_range=False)
    if not is_absolute:
        raise pxt.Error('bboxes_resize_canvas(): requires absolute bounding boxes')

    # Parameter validation
    has_new_dims = new_canvas_width is not None and new_canvas_height is not None
    has_dims = canvas_width is not None and canvas_height is not None
    has_scale = canvas_scale is not None
    has_scale_xy = canvas_scale_x is not None or canvas_scale_y is not None

    if not has_new_dims and not has_scale and not has_scale_xy:
        raise pxt.Error(
            'bboxes_resize_canvas(): requires either all of canvas_width, canvas_height, new_canvas_width, '
            'new_canvas_height, or at least one of canvas_scale, canvas_scale_x, canvas_scale_y to be specified'
        )
    if has_new_dims and not has_dims:
        raise pxt.Error(
            'bboxes_resize_canvas(): new_canvas_width/new_canvas_height also require canvas_width/canvas_height '
            'to be specified'
        )
    if (has_new_dims or has_dims) and (has_scale or has_scale_xy):
        raise pxt.Error(
            'bboxes_resize_canvas(): new_canvas_width/new_canvas_height/canvas_width/canvas_height is mutually '
            'exclusive with canvas_scale/canvas_scale_x/canvas_scale_y'
        )
    if has_scale and has_scale_xy:
        raise pxt.Error('bboxes_resize_canvas(): canvas_scale is mutually exclusive with canvas_scale_x/canvas_scale_y')

    if new_canvas_width is not None and new_canvas_width <= 0:
        raise pxt.Error('bboxes_resize_canvas(): new_canvas_width must be positive')
    if new_canvas_height is not None and new_canvas_height <= 0:
        raise pxt.Error('bboxes_resize_canvas(): new_canvas_height must be positive')
    if canvas_scale is not None and canvas_scale <= 0:
        raise pxt.Error('bboxes_resize_canvas(): canvas_scale must be positive')
    if canvas_scale_x is not None and canvas_scale_x <= 0:
        raise pxt.Error('bboxes_resize_canvas(): canvas_scale_x must be positive')
    if canvas_scale_y is not None and canvas_scale_y <= 0:
        raise pxt.Error('bboxes_resize_canvas(): canvas_scale_y must be positive')
    if canvas_width is not None and canvas_width <= 0:
        raise pxt.Error('bboxes_resize_canvas(): canvas_width must be positive')
    if canvas_height is not None and canvas_height <= 0:
        raise pxt.Error('bboxes_resize_canvas(): canvas_height must be positive')

    # Compute scale factors
    if has_new_dims:
        scale_x = new_canvas_width / canvas_width
        scale_y = new_canvas_height / canvas_height
    elif has_scale:
        scale_x = scale_y = canvas_scale
    else:
        scale_x = canvas_scale_x if canvas_scale_x is not None else 1.0
        scale_y = canvas_scale_y if canvas_scale_y is not None else 1.0

    arr = np.array(bboxes, dtype=np.float64)
    assert arr.ndim == 2 and arr.shape[1] == 4

    # Detect degenerate boxes
    c0, c1, c2, c3 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if format == 'xyxy':
        valid = (c2 > c0) & (c3 > c1)
    elif format in ('xywh', 'cxcywh'):
        valid = (c2 > 0) & (c3 > 0)
    else:
        raise pxt.Error(f'Invalid format: {format!r}')

    orig: np.ndarray | None = None
    if not valid.all():
        orig = arr.copy()

    # Scale x-related (c0, c2) and y-related (c1, c3) columns
    arr[:, 0] *= scale_x
    arr[:, 1] *= scale_y
    arr[:, 2] *= scale_x
    arr[:, 3] *= scale_y

    # Round absolute coordinates
    result = np.floor(arr + 0.5).astype(int)

    # Restore degenerate boxes
    if orig is not None:
        result[~valid] = orig[~valid]

    return result.tolist()


def _get_contours(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Get contour mask with specified thickness."""
    assert mask.dtype == bool
    # find interior pixels: those with all 8 neighbors in the mask (8: include diagonals)
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    boundaries = mask & ~interior

    for _ in range(thickness - 1):
        # binary dilation to all 8 neighbors
        padded = np.pad(boundaries, 1, mode='constant', constant_values=False)
        boundaries = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )

    return boundaries


@pxt.udf
def overlay_segmentation(
    img: PIL.Image.Image,
    segmentation: pxt.Array[(None, None), np.int32],
    *,
    alpha: float = 0.5,
    background: int = 0,
    segment_colors: list[str] | None = None,
    draw_contours: bool = True,
    contour_thickness: int = 1,
) -> PIL.Image.Image:
    """
    Overlays a colored segmentation map on an image.

    Colors can be specified as common HTML color names (e.g., 'red') supported by PIL's
    [`ImageColor`](https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#imagecolor-module) module or as
    RGB hex codes (e.g., '#FF0000').

    If no colors are specified, this function randomly assigns each segment a specific color based on a hash of its id.

    Args:
        img: Input image.
        segmentation: 2D array of the same shape as `img` where each pixel value is a segment id.
        alpha: Blend factor for the overlay (0.0 = only original image, 1.0 = only segmentation colors).
        background: Segment id to treat as background (not overlaid with color, showing the original
            image through).
        segment_colors: List of colors, one per segment id. If the list is shorter than the number of segments, the
            remaining segments will be assigned colors automatically.
        draw_contours: If True, draw contours around each segment with full opacity.
        contour_thickness: Thickness of the contour lines in pixels.

    Returns:
        The image with the colored segmentation overlay.
    """

    if segmentation.shape != (img.height, img.width):
        raise ValueError(
            f'Segmentation shape {segmentation.shape} does not match image dimensions ({img.height}, {img.width})'
        )
    segment_ids = sorted(int(sid) for sid in np.unique(segmentation) if sid != background)
    if segment_colors is None:
        segment_colors = []
    missing_ids = segment_ids[len(segment_colors) :]
    auto_colors = __create_label_colors(missing_ids)
    color_map = {**auto_colors, **dict(zip(segment_ids, segment_colors))}

    segment_alpha = int(alpha * 255)

    overlay_array = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    segment_colors = {id: PIL.ImageColor.getrgb(color_map[id]) for id in segment_ids}
    for segment_id in segment_ids:
        rgb = segment_colors[segment_id]
        mask = segmentation == segment_id
        overlay_array[mask] = (*rgb, segment_alpha)
        if draw_contours:
            contour_mask = _get_contours(mask, contour_thickness)
            overlay_array[contour_mask] = (*rgb, 255)

    overlay = PIL.Image.fromarray(overlay_array, mode='RGBA')
    img_rgba = img.convert('RGBA')
    result = PIL.Image.alpha_composite(img_rgba, overlay)
    return result.convert('RGB')


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

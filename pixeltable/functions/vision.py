"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for Computer Vision.

Example:
```python
import pixeltable as pxt
from pixeltable.functions import vision as pxtv

t = pxt.get_table(...)
t.select(pxtv.draw_bounding_boxes(t.img, boxes=t.boxes, label=t.labels)).collect()
```
"""

import colorsys
import hashlib
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import PIL.Image

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
            'tp': list[int],  # List of 1's and 0's indicating true positives for each
                              # predicted bounding box of this class
            'fp': list[int],  # List of 1's and 0's indicating false positives for each
                              # predicted bounding box of this class; `fp[n] == 1 - tp[n]`
            'scores': list[float],  # List of predicted scores for each bounding box of this class
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
def draw_bounding_boxes(
    img: PIL.Image.Image,
    boxes: list[list[int]],
    labels: Optional[list[Any]] = None,
    color: Optional[str] = None,
    box_colors: Optional[list[str]] = None,
    fill: bool = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> PIL.Image.Image:
    """
    Draws bounding boxes on the given image.

    Labels can be any type that supports `str()` and is hashable (e.g., strings, ints, etc.).

    Colors can be specified as common HTML color names (e.g., 'red') supported by PIL's
    [`ImageColor`](https://pillow.readthedocs.io/en/stable/reference/ImageColor.html#imagecolor-module) module or as
    RGB hex codes (e.g., '#FF0000').

    If no colors are specified, this function randomly assigns each label a specific color based on a hash of the label.

    Args:
        img: The image on which to draw the bounding boxes.
        boxes: List of bounding boxes, each represented as [xmin, ymin, xmax, ymax].
        labels: List of labels for each bounding box.
        color: Single color to be used for all bounding boxes and labels.
        box_colors: List of colors, one per bounding box.
        fill: Whether to fill the bounding boxes with color.
        width: Width of the bounding box borders.
        font: Name of a system font or path to a TrueType font file, as required by
            [`PIL.ImageFont.truetype()`](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.truetype).
            If `None`, uses the default provided by
            [`PIL.ImageFont.load_default()`](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.load_default).
        font_size: Size of the font used for labels in points. Only used in conjunction with non-`None` `font` argument.

    Returns:
        The image with bounding boxes drawn on it.
    """
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

    from PIL import ImageColor, ImageDraw, ImageFont

    # set default font if not provided
    txt_font: Union[ImageFont.ImageFont, ImageFont.FreeTypeFont] = (
        ImageFont.load_default() if font is None else ImageFont.truetype(font=font, size=font_size or 10)
    )

    img_to_draw = img.copy()
    draw = ImageDraw.Draw(img_to_draw, 'RGBA' if fill else 'RGB')

    # Draw bounding boxes
    for i, bbox in enumerate(boxes):
        # determine color for the current box and label
        color = box_colors[i % len(box_colors)]

        if fill:
            rgb_color = ImageColor.getrgb(color)
            fill_color = (*rgb_color, 100)  # semi-transparent
            draw.rectangle(bbox, outline=color, width=width, fill=fill_color)  # type: ignore[arg-type]
        else:
            draw.rectangle(bbox, outline=color, width=width)  # type: ignore[arg-type]

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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

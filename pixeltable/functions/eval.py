from typing import List, Tuple, Dict
from collections import defaultdict
import sys

import numpy as np

import pixeltable.type_system as ts
import pixeltable.func as func


# TODO: figure out a better submodule structure

# the following function has been adapted from MMEval
# (sources at https://github.com/open-mmlab/mmeval)
# Copyright (c) OpenMMLab. All rights reserved.
def calculate_bboxes_area(bboxes: np.ndarray) -> np.ndarray:
    """Calculate area of bounding boxes.

    Args:
        bboxes (numpy.ndarray): The bboxes with shape (n, 4) or (4, ) in 'xyxy' format.
     Returns:
        numpy.ndarray: The area of bboxes.
    """
    bboxes_w = (bboxes[..., 2] - bboxes[..., 0])
    bboxes_h = (bboxes[..., 3] - bboxes[..., 1])
    areas = bboxes_w * bboxes_h
    return areas

# the following function has been adapted from MMEval
# (sources at https://github.com/open-mmlab/mmeval)
# Copyright (c) OpenMMLab. All rights reserved.
def calculate_overlaps(bboxes1: np.ndarray, bboxes2: np.ndarray) -> np.ndarray:
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
    area1 = calculate_bboxes_area(bboxes1)
    area2 = calculate_bboxes_area(bboxes2)
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
def calculate_image_tpfp(
    pred_bboxes: np.ndarray, pred_scores: np.ndarray, gt_bboxes: np.ndarray, min_iou: float
) -> Tuple[np.ndarray, np.ndarray]:
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

        - tp (numpy.ndarray): Shape (N,),
          the true positive flag of each predicted bbox on this image.
        - fp (numpy.ndarray): Shape (N,),
          the false positive flag of each predicted bbox on this image.
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
    ious = calculate_overlaps(pred_bboxes, gt_bboxes)
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

@func.udf(
    return_type=ts.JsonType(nullable=False),
    param_types=[
        ts.JsonType(nullable=False),
        ts.JsonType(nullable=False),
        ts.JsonType(nullable=False),
        ts.JsonType(nullable=False),
        ts.JsonType(nullable=False)
    ])
def eval_detections(
        pred_bboxes: List[List[int]], pred_labels: List[int], pred_scores: List[float],
        gt_bboxes: List[List[int]], gt_labels: List[int]
) -> Dict:
    class_idxs = list(set(pred_labels + gt_labels))
    result: List[Dict] = []
    pred_bboxes_arr = np.asarray(pred_bboxes)
    pred_classes_arr = np.asarray(pred_labels)
    pred_scores_arr = np.asarray(pred_scores)
    gt_bboxes_arr = np.asarray(gt_bboxes)
    gt_classes_arr = np.asarray(gt_labels)
    for class_idx in class_idxs:
        pred_filter = pred_classes_arr == class_idx
        gt_filter = gt_classes_arr == class_idx
        class_pred_scores = pred_scores_arr[pred_filter]
        tp, fp = calculate_image_tpfp(
            pred_bboxes_arr[pred_filter], class_pred_scores, gt_bboxes_arr[gt_filter], [0.5])
        ordered_class_pred_scores = -np.sort(-class_pred_scores)
        result.append({
            'min_iou': 0.5, 'class': class_idx, 'tp': tp.tolist(), 'fp': fp.tolist(),
            'scores': ordered_class_pred_scores.tolist(), 'num_gts': gt_filter.sum().item(),
        })
    return result

@func.uda(
    update_types=[ts.JsonType()], value_type=ts.JsonType(), allows_std_agg=True, allows_window=False)
class mean_ap(func.Aggregator):
    def __init__(self):
        self.class_tpfp: Dict[int, List[Dict]] = defaultdict(list)

    def update(self, eval_dicts: List[Dict]) -> None:
        for eval_dict in eval_dicts:
            class_idx = eval_dict['class']
            self.class_tpfp[class_idx].append(eval_dict)

    def value(self) -> Dict:
        eps = np.finfo(np.float32).eps
        result: Dict[int, float] = {}
        for class_idx, tpfp in self.class_tpfp.items():
            a1 = [x['tp'] for x in tpfp]
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

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.vision import bboxes_draw, bboxes_resize, eval_detections, mean_ap, overlay_segmentation
from pixeltable.functions.yolox import yolox

from ..utils import get_image_files, get_video_files, skip_test_if_not_installed


class TestVision:
    def test_eval(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')

        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        # create frame view
        v = pxt.create_view('test_view', video_t, iterator=frame_iterator(video_t.video, fps=1))

        files = get_video_files()
        video_t.insert(video=files[-1])
        v.add_computed_column(frame_s=v.frame.resize([640, 480]))
        v.add_computed_column(detections_a=yolox(v.frame_s, model_id='yolox_nano'))
        v.add_computed_column(detections_b=yolox(v.frame_s, model_id='yolox_s'))
        v.add_computed_column(gt=yolox(v.frame_s, model_id='yolox_m'))

        _ = v.select(
            eval_detections(
                v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels
            )
        ).show()
        v.add_computed_column(
            eval_a=eval_detections(
                v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels
            )
        )
        v.add_computed_column(
            eval_b=eval_detections(
                v.detections_b.bboxes,
                v.detections_b.labels,
                v.detections_b.scores,
                v.gt.bboxes,
                v.gt.labels,
                min_iou=0.8,
            )
        )
        _ = v.select(mean_ap(v.eval_a)).show()[0, 0]
        _ = v.select(mean_ap(v.eval_b)).show()[0, 0]

        _ = v.select(
            bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=True)
        ).collect()

    def test_draw_bounding_boxes(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')

        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        # create frame view
        v = pxt.create_view('test_view', video_t, iterator=frame_iterator(video_t.video, fps=1))

        files = get_video_files()
        video_t.insert(video=files[-1])
        v.add_computed_column(frame_s=v.frame.resize([640, 480]))
        v.add_computed_column(detections_a=yolox(v.frame_s, model_id='yolox_nano'))

        # default label colors
        _ = v.select(
            bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=True)
        ).collect()
        _ = v.select(
            bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=False, width=3)
        ).collect()
        for color in ['red', '#FF0000FF']:
            for alpha in [None, 0.5]:
                for fill_alpha in [None, 0.3]:
                    _ = v.select(
                        bboxes_draw(
                            v.frame_s,
                            boxes=v.detections_a.bboxes,
                            labels=v.detections_a.labels,
                            color=color,
                            alpha=alpha,
                            fill=fill_alpha is not None,
                            fill_alpha=fill_alpha,
                        )
                    ).collect()

                # explicit box colors
                num_boxes = len(v.where(v.pos == 0).select(v.detections_a.bboxes).collect()[0, 0])
                box_colors = [color] * num_boxes
                _ = (
                    v.where(v.pos == 0)
                    .select(
                        bboxes_draw(
                            v.frame_s,
                            boxes=v.detections_a.bboxes,
                            labels=v.detections_a.labels,
                            box_colors=box_colors,
                            alpha=alpha,
                            fill=fill_alpha is not None,
                            fill_alpha=fill_alpha,
                        )
                    )
                    .collect()
                )

        with pytest.raises(pxt.Error) as exc_info:
            # multiple color specifications
            _ = v.select(
                bboxes_draw(
                    v.frame_s,
                    boxes=v.detections_a.bboxes,
                    labels=v.detections_a.labels,
                    box_colors=['red', 'green'],
                    color='red',
                )
            ).collect()
        assert 'only one of' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # labels don't match boxes
            _ = v.select(bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, labels=[2])).collect()
        assert 'number of boxes and labels must match' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # box_colors don't match boxes
            _ = v.select(bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, box_colors=['red'])).collect()
        assert 'number of boxes and box colors must match' in str(exc_info.value).lower()

        # TODO: test font and font_size parameters in a system-independent way

    def test_bboxes_resize(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_tbl', {'bboxes': pxt.Json, 'id': pxt.Int})

        # Box at (100,100)-(200,300): w=100, h=200, center=(150,200)
        t.insert(
            [
                {'bboxes': [[100, 100, 200, 300]], 'id': 1},
                # Box at (0,0)-(400,200): w=400, h=200, center=(200,100)
                {'bboxes': [[0, 0, 400, 200]], 'id': 2},
            ]
        )

        # --- resize to target width (xyxy) ---
        # Box1: w=100,h=200 -> target_w=50, scale=0.5, new_h=100, cx=150,cy=200
        #   -> [125, 150, 175, 250]
        # Box2: w=400,h=200 -> target_w=50, scale=0.125, new_h=25, cx=200,cy=100
        #   -> [175, 87.5, 225, 112.5]
        result = t.order_by(t.id).select(bboxes_resize(t.bboxes, 'xyxy', width=50)).collect()
        r1 = result[0, 0]
        assert r1 == [[125.0, 150.0, 175.0, 250.0]]
        r2 = result[1, 0]
        assert r2 == [[175.0, 87.5, 225.0, 112.5]]

        # --- resize to target height (xyxy) ---
        # Box1: w=100,h=200 -> target_h=100, scale=0.5, new_w=50, cx=150,cy=200
        #   -> [125, 150, 175, 250]
        result = t.where(t.id == 1).select(bboxes_resize(t.bboxes, 'xyxy', height=100)).collect()
        assert result[0, 0] == [[125.0, 150.0, 175.0, 250.0]]

        # --- resize to target width_f (relative coords, xyxy) ---
        t2 = pxt.create_table('bbox_tbl2', {'bboxes': pxt.Json})
        # Box in relative coords: (0.1, 0.2, 0.5, 0.8) -> w=0.4, h=0.6, cx=0.3, cy=0.5
        t2.insert([{'bboxes': [[0.1, 0.2, 0.5, 0.8]]}])
        # target_w_f=0.2, scale=0.5, new_h=0.3
        # -> [0.2, 0.35, 0.4, 0.65]
        result = t2.select(bboxes_resize(t2.bboxes, 'xyxy', width_f=0.2)).collect()
        r = result[0, 0]
        np.testing.assert_allclose(r, [[0.2, 0.35, 0.4, 0.65]])

        # --- resize to target height_f (relative coords, xywh) ---
        # Box in xywh: (0.1, 0.2, 0.4, 0.6) -> cx=0.3, cy=0.5, w=0.4, h=0.6
        # target_h_f=0.3, scale=0.5, new_w=0.2
        # -> xywh: [0.2, 0.35, 0.2, 0.3]
        t3 = pxt.create_table('bbox_tbl3', {'bboxes': pxt.Json})
        t3.insert([{'bboxes': [[0.1, 0.2, 0.4, 0.6]]}])
        result = t3.select(bboxes_resize(t3.bboxes, 'xywh', height_f=0.3)).collect()
        np.testing.assert_allclose(result[0, 0], [[0.2, 0.35, 0.2, 0.3]])

        # --- resize with cxcywh format ---
        # cx=150, cy=200, w=100, h=200 -> target_w=50, scale=0.5, new_h=100
        # -> cxcywh: [150, 200, 50, 100]
        result = t.where(t.id == 1).select(bboxes_resize([[150, 200, 100, 200]], 'cxcywh', width=50)).collect()
        assert result[0, 0] == [[150.0, 200.0, 50.0, 100.0]]

        # --- aspect ratio crop (xyxy) ---
        # Box2: w=400, h=200, aspect=2.0, target 1:1 (aspect=1.0)
        # crop: too_wide (2>1), new_w=200*1=200, new_h stays 200, cx=200, cy=100
        # -> [100, 0, 300, 200]
        result = t.where(t.id == 2).select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1', aspect_mode='crop')).collect()
        assert result[0, 0] == [[100.0, 0.0, 300.0, 200.0]]

        # --- aspect ratio pad (xyxy) ---
        # Box2: w=400, h=200, aspect=2.0, target 1:1
        # pad: too_wide (2>1), new_w stays 400, new_h=400/1=400, cx=200, cy=100
        # -> [0, -100, 400, 300]
        result = t.where(t.id == 2).select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1', aspect_mode='pad')).collect()
        assert result[0, 0] == [[0.0, -100.0, 400.0, 300.0]]

        # --- aspect_f ---
        # Same as aspect='1:1' but using aspect_f=1.0
        result = t.where(t.id == 2).select(bboxes_resize(t.bboxes, 'xyxy', aspect_f=1.0, aspect_mode='crop')).collect()
        assert result[0, 0] == [[100.0, 0.0, 300.0, 200.0]]

        # --- error: no size parameter ---
        with pytest.raises(pxt.Error):
            t.select(bboxes_resize(t.bboxes, 'xyxy')).collect()

        # --- error: multiple size parameters ---
        with pytest.raises(pxt.Error):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, height=50)).collect()

        # --- error: aspect without aspect_mode ---
        with pytest.raises(pxt.Error):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1')).collect()

        # --- error: aspect_mode without aspect ---
        with pytest.raises(pxt.Error):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, aspect_mode='crop')).collect()

    def test_overlay_segmentation(self, uses_db: None) -> None:
        skip_test_if_not_installed('transformers')

        from pixeltable.functions.huggingface import detr_for_segmentation

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(
            segmentation=detr_for_segmentation(t.img, model_id='facebook/detr-resnet-50-panoptic', threshold=0.5)
        )
        image_files = get_image_files()[:3]
        t.insert({'img': f} for f in image_files)

        segmentation_map = t.segmentation.segmentation.astype(pxt.Array[(None, None), np.int32])  # type: ignore[misc]
        _ = t.select(overlay_segmentation(t.img, segmentation_map)).collect()

        # test non-defaults
        label_colors = ['red', '#00FF00']
        _ = t.select(
            overlay_segmentation(t.img, segmentation_map, alpha=0.3, background=1, segment_colors=label_colors)
        ).collect()

        # test draw_contours
        _ = t.select(overlay_segmentation(t.img, segmentation_map, draw_contours=True, contour_thickness=2)).collect()

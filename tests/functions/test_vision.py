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
        t = pxt.create_table('bbox_tbl', {'id': pxt.Int})
        t.insert([{'id': 1}])

        def to_bbox(cx, cy, w, h, fmt):
            if fmt == 'xyxy':
                return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
            elif fmt == 'xywh':
                return [cx - w / 2, cy - h / 2, w, h]
            else:  # cxcywh
                return [cx, cy, w, h]

        # (input_cxcywh, resize_kwargs, expected_cxcywh, use_float)
        test_cases = [
            ((150, 200, 100, 200), {'width': 50}, (150, 200, 50, 100), False),
            ((150, 200, 100, 200), {'height': 100}, (150, 200, 50, 100), False),
            ((200, 100, 400, 200), {'aspect': '1:1', 'aspect_mode': 'crop'}, (200, 100, 200, 200), False),
            ((200, 100, 400, 200), {'aspect': '1:1', 'aspect_mode': 'pad'}, (200, 100, 400, 400), False),
            ((200, 100, 400, 200), {'aspect_f': 1.0, 'aspect_mode': 'crop'}, (200, 100, 200, 200), False),
            ((0.3, 0.5, 0.4, 0.6), {'width_f': 0.2}, (0.3, 0.5, 0.2, 0.3), True),
            ((0.3, 0.5, 0.4, 0.6), {'height_f': 0.3}, (0.3, 0.5, 0.2, 0.3), True),
        ]

        formats = ['xyxy', 'xywh', 'cxcywh']
        for fmt in formats:
            for input_cxcywh, kwargs, expected_cxcywh, use_float in test_cases:
                input_bbox = to_bbox(*input_cxcywh, fmt)
                if not use_float:
                    input_bbox = [int(x) for x in input_bbox]
                expected_bbox = to_bbox(*expected_cxcywh, fmt)
                result = t.select(bboxes_resize([input_bbox], fmt, **kwargs)).collect()
                np.testing.assert_allclose(
                    result[0, 0], [expected_bbox], err_msg=f'format={fmt}, kwargs={kwargs}'
                )

    def test_bboxes_resize_errors(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_tbl', {'bboxes': pxt.Json})
        t.insert([{'bboxes': [[100, 100, 200, 300]]}])

        # no size parameter
        with pytest.raises(pxt.Error, match='Exactly one of'):
            t.select(bboxes_resize(t.bboxes, 'xyxy')).collect()

        # multiple size parameters
        with pytest.raises(pxt.Error, match='Exactly one of'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, height=50)).collect()

        # width + width_f
        with pytest.raises(pxt.Error, match='Only one of width or width_f'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, width_f=0.5)).collect()

        # height + height_f
        with pytest.raises(pxt.Error, match='Only one of height or height_f'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', height=50, height_f=0.5)).collect()

        # aspect + aspect_f
        with pytest.raises(pxt.Error, match='Only one of aspect or aspect_f'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1', aspect_f=1.0, aspect_mode='crop')).collect()

        # invalid aspect ratio string
        with pytest.raises(pxt.Error, match='Invalid aspect ratio'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='bad', aspect_mode='crop')).collect()

        # aspect without aspect_mode
        with pytest.raises(pxt.Error, match='aspect_mode.*required'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1')).collect()

        # aspect_mode without aspect
        with pytest.raises(pxt.Error, match='aspect_mode is only valid'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, aspect_mode='crop')).collect()

        # mixed int/float coordinates
        t_mixed = pxt.create_table('bbox_mixed', {'bboxes': pxt.Json})
        t_mixed.insert([{'bboxes': [[100, 100.0, 200, 300]]}])
        with pytest.raises(pxt.Error, match='either all int or all float'):
            t_mixed.select(bboxes_resize(t_mixed.bboxes, 'xyxy', width=50)).collect()

        # wrong number of coordinates
        t_bad = pxt.create_table('bbox_bad', {'bboxes': pxt.Json})
        t_bad.insert([{'bboxes': [[100, 100, 200]]}])
        with pytest.raises(pxt.Error, match='exactly 4 coordinates'):
            t_bad.select(bboxes_resize(t_bad.bboxes, 'xyxy', width=50)).collect()

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

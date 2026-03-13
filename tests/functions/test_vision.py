import math
from typing import Any

import numpy as np
import pytest

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.vision import (
    bboxes_clip_to_canvas,
    bboxes_convert,
    bboxes_draw,
    bboxes_pad,
    bboxes_resize,
    bboxes_scale,
    eval_detections,
    mean_ap,
    overlay_segmentation,
)

from ..utils import get_image_files, get_video_files, skip_test_if_not_installed, validate_update_status


class TestVision:
    def test_eval(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')
        from pixeltable.functions.yolox import yolox

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

    def test_bboxes_draw(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')
        from pixeltable.functions.yolox import yolox

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

        with pytest.raises(pxt.Error, match='Only one of'):
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

        with pytest.raises(pxt.Error, match='Number of boxes and labels must match'):
            # labels don't match boxes
            _ = v.select(bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, labels=[2])).collect()

        with pytest.raises(pxt.Error, match='Number of boxes and box colors must match'):
            # box_colors don't match boxes
            _ = v.select(bboxes_draw(v.frame_s, boxes=v.detections_a.bboxes, box_colors=['red'])).collect()

        # TODO: test font and font_size parameters in a system-independent way

    def test_bboxes_resize(self, uses_db: None) -> None:
        # absolute coordinates, in cxcywh format
        abs_boxes: list[tuple[int, int, int, int]] = [
            (150, 200, 100, 200),
            (200, 100, 400, 200),
            (50, 50, 40, 60),
            (300, 300, 200, 100),
            (100, 100, 80, 80),
        ]
        # relative coordinates, in cxcywh format
        rel_boxes: list[tuple[float, float, float, float]] = [
            (0.3, 0.5, 0.4, 0.6),
            (0.5, 0.5, 0.2, 0.2),
            (0.7, 0.3, 0.1, 0.4),
            (0.2, 0.8, 0.3, 0.1),
            (0.9, 0.1, 0.1, 0.1),
        ]

        formats = ['xyxy', 'xywh', 'cxcywh']
        # test case for each parameter against each format
        for fmt in formats:
            # corner case: empty list
            t = pxt.create_table('bbox_empty', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': []}]), expected_rows=1)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, width=50)).collect()
            assert res['out'][0] == []
            pxt.drop_table(t)

            input_bboxes = [convert_fmt(*b, fmt) for b in abs_boxes]
            t = pxt.create_table('bbox_abs', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

            # width
            res = t.select(out=bboxes_resize(t.bboxes, fmt, width=50)).collect()
            assert all(get_w(b, fmt) == 50 for b in res['out'][0])
            assert all(
                get_aspect(b1, fmt) == pytest.approx(get_aspect(b2, fmt)) for b1, b2 in zip(input_bboxes, res['out'][0])
            )

            # height
            res = t.select(out=bboxes_resize(t.bboxes, fmt, height=100)).collect()
            assert all(get_h(b, fmt) == 100 for b in res['out'][0])
            assert all(
                get_aspect(b1, fmt) == pytest.approx(get_aspect(b2, fmt), rel=0.01)
                for b1, b2 in zip(input_bboxes, res['out'][0])
            )

            # aspect 16:9 crop
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect='16:9', aspect_mode='crop')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(16 / 9, rel=0.1) for b in res['out'][0])
            assert all(crop_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect 9:16 pad
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect='9:16', aspect_mode='pad')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(9 / 16, rel=0.1) for b in res['out'][0])
            assert all(pad_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect (float) 16/9 crop
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect=16 / 9, aspect_mode='crop')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(16 / 9, rel=0.1) for b in res['out'][0])
            assert all(crop_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect (float) 9/16 pad
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect=9 / 16, aspect_mode='pad')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(9 / 16, rel=0.1) for b in res['out'][0])
            assert all(pad_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            pxt.drop_table(t)

            input_bboxes = [convert_fmt(*b, fmt) for b in rel_boxes]
            t = pxt.create_table('bbox_rel', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

            # width (float/relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, width=0.2)).collect()
            assert all(get_w(b, fmt) == pytest.approx(0.2) for b in res['out'][0])
            assert all(
                get_aspect(b1, fmt) == pytest.approx(get_aspect(b2, fmt)) for b1, b2 in zip(input_bboxes, res['out'][0])
            )

            # height (float/relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, height=0.3)).collect()
            assert all(get_h(b, fmt) == pytest.approx(0.3) for b in res['out'][0])
            assert all(
                get_aspect(b1, fmt) == pytest.approx(get_aspect(b2, fmt)) for b1, b2 in zip(input_bboxes, res['out'][0])
            )

            # aspect 16:9 crop (relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect='16:9', aspect_mode='crop')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(16 / 9) for b in res['out'][0])
            assert all(crop_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect 9:16 pad (relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect='9:16', aspect_mode='pad')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(9 / 16) for b in res['out'][0])
            assert all(pad_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect (float) 16/9 crop (relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect=16 / 9, aspect_mode='crop')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(16 / 9) for b in res['out'][0])
            assert all(crop_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            # aspect (float) 9/16 pad (relative)
            res = t.select(out=bboxes_resize(t.bboxes, fmt, aspect=9 / 16, aspect_mode='pad')).collect()
            assert all(get_aspect(b, fmt) == pytest.approx(9 / 16) for b in res['out'][0])
            assert all(pad_invariant(b_in, b_out, fmt) for b_in, b_out in zip(input_bboxes, res['out'][0]))

            pxt.drop_table(t)

    def test_bboxes_resize_errors(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_tbl', {'bboxes': pxt.Json})
        t.insert([{'bboxes': [[100, 100, 200, 300]]}])

        # no size parameter
        with pytest.raises(pxt.Error, match='Exactly one of'):
            t.select(bboxes_resize(t.bboxes, 'xyxy')).collect()

        # invalid format
        with pytest.raises(pxt.Error, match='Invalid format'):
            t.select(bboxes_resize(t.bboxes, 'coco', width=50)).collect()

        # multiple size parameters (int)
        with pytest.raises(pxt.Error, match='Exactly one of'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, height=50)).collect()

        # invalid aspect mode (float aspect)
        with pytest.raises(pxt.Error, match='Invalid aspect_mode'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect=16 / 9, aspect_mode='other')).collect()

        # invalid aspect ratio string
        with pytest.raises(pxt.Error, match='Invalid aspect ratio'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='bad', aspect_mode='crop')).collect()

        # aspect without aspect_mode
        with pytest.raises(pxt.Error, match=r'aspect_mode.*required'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='1:1')).collect()

        # aspect_mode without aspect
        with pytest.raises(pxt.Error, match='aspect_mode is only valid'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=50, aspect_mode='crop')).collect()

        # float width with absolute boxes (dispatches to float overload → _bboxes_resize(width_f=...))
        with pytest.raises(pxt.Error, match='width/height require relative'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=0.5)).collect()

        # float height with absolute boxes
        with pytest.raises(pxt.Error, match='width/height require relative'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', height=0.5)).collect()

        # int width with relative boxes
        t_rel = pxt.create_table('bbox_rel', {'bboxes': pxt.Json})
        t_rel.insert([{'bboxes': [[0.1, 0.2, 0.3, 0.4]]}])
        with pytest.raises(pxt.Error, match='width/height require absolute'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', width=50)).collect()

        # int height with relative boxes
        with pytest.raises(pxt.Error, match='width/height require absolute'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', height=50)).collect()

        # zero/negative width (int)
        with pytest.raises(pxt.Error, match='width must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=0)).collect()
        with pytest.raises(pxt.Error, match='width must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', width=-10)).collect()

        # zero/negative height (int)
        with pytest.raises(pxt.Error, match='height must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', height=0)).collect()
        with pytest.raises(pxt.Error, match='height must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', height=-10)).collect()

        # zero/negative width (float)
        with pytest.raises(pxt.Error, match='width must be positive'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', width=0.0)).collect()
        with pytest.raises(pxt.Error, match='width must be positive'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', width=-0.1)).collect()

        # zero/negative height (float)
        with pytest.raises(pxt.Error, match='height must be positive'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', height=0.0)).collect()
        with pytest.raises(pxt.Error, match='height must be positive'):
            t_rel.select(bboxes_resize(t_rel.bboxes, 'xyxy', height=-0.1)).collect()

        # zero/negative aspect (float)
        with pytest.raises(pxt.Error, match='aspect must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect=0.0, aspect_mode='crop')).collect()
        with pytest.raises(pxt.Error, match='aspect must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect=-1.0, aspect_mode='crop')).collect()

        # aspect string with zero component
        with pytest.raises(pxt.Error, match='width and height must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='0:9', aspect_mode='crop')).collect()
        with pytest.raises(pxt.Error, match='width and height must be positive'):
            t.select(bboxes_resize(t.bboxes, 'xyxy', aspect='9:0', aspect_mode='crop')).collect()

    def test_bboxes_resize_degenerate(self, uses_db: None) -> None:
        degenerate_boxes = [
            [10, 20, 10, 40],  # zero width (xyxy)
            [10, 20, 30, 20],  # zero height (xyxy)
            [10, 20, 10, 20],  # zero width and height (xyxy)
            [30, 40, 10, 20],  # negative width and height (xyxy, x2<x1, y2<y1)
        ]
        t = pxt.create_table('degenerate', {'bboxes': pxt.Json})
        t.insert([{'bboxes': degenerate_boxes}])
        res = t.select(out=bboxes_resize(t.bboxes, 'xyxy', width=50)).collect()
        assert res['out'][0] == degenerate_boxes  # all passed through unchanged

        self._test_bbox_validation(t, bboxes_resize(t.bboxes, 'xyxy', width=50))

    def test_bboxes_scale(self, uses_db: None) -> None:
        # absolute coordinates, in cxcywh format
        abs_boxes: list[tuple[int, int, int, int]] = [
            (150, 200, 100, 200),
            (200, 100, 400, 200),
            (50, 50, 40, 60),
            (300, 300, 200, 100),
            (100, 100, 80, 80),
        ]
        # relative coordinates, in cxcywh format
        rel_boxes: list[tuple[float, float, float, float]] = [
            (0.3, 0.5, 0.4, 0.6),
            (0.5, 0.5, 0.2, 0.2),
            (0.7, 0.3, 0.1, 0.4),
            (0.2, 0.8, 0.3, 0.1),
            (0.9, 0.1, 0.1, 0.1),
        ]

        formats = ['xyxy', 'xywh', 'cxcywh']
        for fmt in formats:
            # corner case: empty list
            t = pxt.create_table('bbox_empty', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': []}]), expected_rows=1)
            res = t.select(out=bboxes_scale(t.bboxes, fmt, factor=2.0)).collect()
            assert res['out'][0] == []
            pxt.drop_table(t)

            # absolute coordinates

            input_bboxes = [convert_fmt(*b, fmt) for b in abs_boxes]
            t = pxt.create_table('bbox_abs', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

            # factor=2.0: doubles both w and h, center stays same
            res = t.select(out=bboxes_scale(t.bboxes, fmt, factor=2.0)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) * 2, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) * 2, abs=1)

            # x_factor=2.0: doubles w only
            res = t.select(out=bboxes_scale(t.bboxes, fmt, x_factor=2.0)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) * 2, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt), abs=1)

            # y_factor=0.5: halves h only
            res = t.select(out=bboxes_scale(t.bboxes, fmt, y_factor=0.5)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt), abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) * 0.5, abs=1)

            # x_factor=2.0, y_factor=3.0: scales independently
            res = t.select(out=bboxes_scale(t.bboxes, fmt, x_factor=2.0, y_factor=3.0)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) * 2, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) * 3, abs=1)

            pxt.drop_table(t)

            # relative coordinates

            input_bboxes = [convert_fmt(*b, fmt) for b in rel_boxes]
            t = pxt.create_table('bbox_rel', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

            # factor=2.0
            res = t.select(out=bboxes_scale(t.bboxes, fmt, factor=2.0)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) * 2)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) * 2)

            # x_factor=0.5
            res = t.select(out=bboxes_scale(t.bboxes, fmt, x_factor=0.5)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) * 0.5)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt))

            # y_factor=3.0
            res = t.select(out=bboxes_scale(t.bboxes, fmt, y_factor=3.0)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt))
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) * 3)

            pxt.drop_table(t)

    def test_bboxes_scale_errors(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_tbl', {'bboxes': pxt.Json})
        t.insert([{'bboxes': [[100, 100, 200, 300]]}])

        # no factors specified
        with pytest.raises(pxt.Error, match='at least one of'):
            t.select(bboxes_scale(t.bboxes, 'xyxy')).collect()

        # factor with x_factor
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', factor=2.0, x_factor=1.5)).collect()

        # factor with y_factor
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', factor=2.0, y_factor=1.5)).collect()

        # factor with both x_factor and y_factor
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', factor=2.0, x_factor=1.5, y_factor=1.5)).collect()

        # zero factor
        with pytest.raises(pxt.Error, match='factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', factor=0.0)).collect()

        # negative factor
        with pytest.raises(pxt.Error, match='factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', factor=-1.0)).collect()

        # zero x_factor
        with pytest.raises(pxt.Error, match='x_factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', x_factor=0.0)).collect()

        # negative x_factor
        with pytest.raises(pxt.Error, match='x_factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', x_factor=-1.0)).collect()

        # zero y_factor
        with pytest.raises(pxt.Error, match='y_factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', y_factor=0.0)).collect()

        # negative y_factor
        with pytest.raises(pxt.Error, match='y_factor must be positive'):
            t.select(bboxes_scale(t.bboxes, 'xyxy', y_factor=-1.0)).collect()

    def test_bboxes_scale_degenerate(self, uses_db: None) -> None:
        degenerate_boxes = [
            [10, 20, 10, 40],  # zero width (xyxy)
            [10, 20, 30, 20],  # zero height (xyxy)
            [10, 20, 10, 20],  # zero width and height (xyxy)
            [30, 40, 10, 20],  # negative width and height (xyxy, x2<x1, y2<y1)
        ]
        t = pxt.create_table('degenerate', {'bboxes': pxt.Json})
        t.insert([{'bboxes': degenerate_boxes}])
        res = t.select(out=bboxes_scale(t.bboxes, 'xyxy', factor=2.0)).collect()
        assert res['out'][0] == degenerate_boxes  # all passed through unchanged

        self._test_bbox_validation(t, bboxes_scale(t.bboxes, 'xyxy', factor=2.0))

    def test_bboxes_pad(self, uses_db: None) -> None:
        # absolute coordinates, in cxcywh format
        abs_boxes: list[tuple[int, int, int, int]] = [
            (150, 200, 100, 200),
            (200, 100, 400, 200),
            (50, 50, 40, 60),
            (300, 300, 200, 100),
            (100, 100, 80, 80),
        ]

        formats = ['xyxy', 'xywh', 'cxcywh']
        for fmt in formats:
            # corner case: empty list
            t = pxt.create_table('bbox_empty', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': []}]), expected_rows=1)
            res = t.select(out=bboxes_pad(t.bboxes, fmt, x=10)).collect()
            assert res['out'][0] == []
            pxt.drop_table(t)

            input_bboxes = [convert_fmt(*b, fmt) for b in abs_boxes]
            t = pxt.create_table('bbox_pad', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

            # symmetric: x=10, y=20 — w grows by 20, h grows by 40, center unchanged
            res = t.select(out=bboxes_pad(t.bboxes, fmt, x=10, y=20)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) + 20, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) + 40, abs=1)

            # asymmetric: left=5, right=15, top=10, bottom=30 — w grows by 20, h grows by 40
            res = t.select(out=bboxes_pad(t.bboxes, fmt, left=5, right=15, top=10, bottom=30)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) + 20, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) + 40, abs=1)

            # single side: left=10 only — w grows by 10
            res = t.select(out=bboxes_pad(t.bboxes, fmt, left=10)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) + 10, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt), abs=1)

            # x only: height unchanged
            res = t.select(out=bboxes_pad(t.bboxes, fmt, x=10)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt) + 20, abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt), abs=1)

            # y only: width unchanged
            res = t.select(out=bboxes_pad(t.bboxes, fmt, y=15)).collect()
            for b_in, b_out in zip(input_bboxes, res['out'][0]):
                assert get_w(b_out, fmt) == pytest.approx(get_w(b_in, fmt), abs=1)
                assert get_h(b_out, fmt) == pytest.approx(get_h(b_in, fmt) + 30, abs=1)

            pxt.drop_table(t)

    def test_bboxes_pad_errors(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_tbl', {'bboxes': pxt.Json})
        t.insert([{'bboxes': [[100, 100, 200, 300]]}])

        # no params specified
        with pytest.raises(pxt.Error, match='at least one padding parameter'):
            t.select(bboxes_pad(t.bboxes, 'xyxy')).collect()

        # x with left
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_pad(t.bboxes, 'xyxy', x=10, left=5)).collect()

        # x with right
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_pad(t.bboxes, 'xyxy', x=10, right=5)).collect()

        # y with top
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_pad(t.bboxes, 'xyxy', y=10, top=5)).collect()

        # y with bottom
        with pytest.raises(pxt.Error, match='mutually exclusive'):
            t.select(bboxes_pad(t.bboxes, 'xyxy', y=10, bottom=5)).collect()

        # negative value
        with pytest.raises(pxt.Error, match='must be >= 0'):
            t.select(bboxes_pad(t.bboxes, 'xyxy', x=-5)).collect()

        # relative bboxes
        t2 = pxt.create_table('bbox_rel', {'bboxes': pxt.Json})
        t2.insert([{'bboxes': [[0.1, 0.2, 0.3, 0.4]]}])
        with pytest.raises(pxt.Error, match='absolute pixel coordinates'):
            t2.select(bboxes_pad(t2.bboxes, 'xyxy', x=10)).collect()

    def test_bboxes_pad_degenerate(self, uses_db: None) -> None:
        degenerate_boxes = [
            [10, 20, 10, 40],  # zero width (xyxy)
            [10, 20, 30, 20],  # zero height (xyxy)
            [10, 20, 10, 20],  # zero width and height (xyxy)
            [30, 40, 10, 20],  # negative width and height (xyxy, x2<x1, y2<y1)
        ]
        t = pxt.create_table('degenerate', {'bboxes': pxt.Json})
        t.insert([{'bboxes': degenerate_boxes}])
        res = t.select(out=bboxes_pad(t.bboxes, 'xyxy', x=10, y=20)).collect()
        assert res['out'][0] == degenerate_boxes  # all passed through unchanged

    def test_bboxes_convert(self, uses_db: None) -> None:
        abs_boxes = [
            (150, 200, 100, 200),
            (200, 100, 400, 200),
            (50, 50, 40, 60),
            (300, 300, 200, 100),
            (100, 100, 80, 80),
            (100, 100, 31, 41),  # odd w and h: exercises rounding in cxcywh conversion
        ]
        rel_boxes = [
            (0.3, 0.5, 0.4, 0.6),
            (0.5, 0.5, 0.2, 0.2),
            (0.7, 0.3, 0.1, 0.4),
            (0.2, 0.8, 0.3, 0.1),
            (0.9, 0.1, 0.1, 0.1),
        ]

        formats = ['xyxy', 'xywh', 'cxcywh']

        for src_fmt in formats:
            # corner case: empty list
            t = pxt.create_table('bbox_empty', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': []}]), expected_rows=1)
            res = t.select(out=bboxes_convert(t.bboxes, src_format=src_fmt, dst_format=src_fmt)).collect()
            assert res['out'][0] == []
            pxt.drop_table(t)

            for boxes in [abs_boxes, rel_boxes]:
                input_bboxes = [convert_fmt(*b, src_fmt) for b in boxes]  # type: ignore
                t = pxt.create_table('convert', {'bboxes': pxt.Json})
                validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

                # identity: same format returns input unchanged
                res = t.select(out=bboxes_convert(t.bboxes, src_format=src_fmt, dst_format=src_fmt)).collect()
                assert res['out'][0] == input_bboxes

                for dst_fmt in formats:
                    if dst_fmt == src_fmt:
                        continue

                    res = t.select(out=bboxes_convert(t.bboxes, src_format=src_fmt, dst_format=dst_fmt)).collect()
                    converted = res['out'][0]

                    # width and height preserved
                    assert all(
                        get_w(b_out, dst_fmt) == pytest.approx(get_w(b_in, src_fmt), abs=1e-9)
                        for b_in, b_out in zip(input_bboxes, converted)
                    )
                    assert all(
                        get_h(b_out, dst_fmt) == pytest.approx(get_h(b_in, src_fmt), abs=1e-9)
                        for b_in, b_out in zip(input_bboxes, converted)
                    )

                    # round-trip: convert back dst -> src, should match original;
                    # cxcywh with odd integer dimensions can shift position by 1
                    # (half-pixel center rounded to int), so relax tolerance
                    uses_cxcywh = 'cxcywh' in (src_fmt, dst_fmt)
                    is_abs = boxes is abs_boxes
                    delta = 1 if (uses_cxcywh and is_abs) else 1e-9
                    res_rt = t.select(
                        out=bboxes_convert(
                            bboxes_convert(t.bboxes, src_format=src_fmt, dst_format=dst_fmt),
                            src_format=dst_fmt,
                            dst_format=src_fmt,
                        )
                    ).collect()
                    assert all(
                        all(v_rt == pytest.approx(v_in, abs=delta) for v_in, v_rt in zip(b_in, b_rt))
                        for b_in, b_rt in zip(input_bboxes, res_rt['out'][0])
                    )

                pxt.drop_table(t)

    def test_bboxes_convert_errors(self, uses_db: None) -> None:
        t = pxt.create_table('convert_err', {'bboxes': pxt.Json})
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='Invalid src_format'):
            t.select(bboxes_convert(t.bboxes, src_format='coco', dst_format='xyxy')).collect()
        with pytest.raises(pxt.Error, match='Invalid dst_format'):
            t.select(bboxes_convert(t.bboxes, src_format='xyxy', dst_format='coco')).collect()
        t.delete()

        self._test_bbox_validation(t, bboxes_convert(t.bboxes, src_format='xyxy', dst_format='xywh'))

    def test_bboxes_clip_to_canvas(self, uses_db: None) -> None:
        # cxcywh format
        # absolute (640x480 canvas)
        abs_boxes: list[tuple[int, int, int, int]] = [
            (150, 200, 100, 200),  # fully inside
            (630, 470, 40, 30),  # partially outside right+bottom, visibility=0.625
            (300, 240, 60, 40),  # fully inside
            (10, 240, 60, 40),  # partially outside left (x1=-20), visibility=0.667
            (320, 540, 40, 40),  # fully outside bottom, visibility=0
        ]
        # relative ([0,1] canvas)
        rel_boxes: list[tuple[float, float, float, float]] = [
            (0.3, 0.5, 0.4, 0.6),  # fully inside
            (0.95, 0.9, 0.2, 0.3),  # partially outside right+bottom, visibility=0.625
            (0.5, 0.5, 0.2, 0.2),  # fully inside
            (0.05, 0.5, 0.3, 0.2),  # partially outside left (x1=-0.1), visibility=0.667
            (0.5, 1.2, 0.2, 0.1),  # fully outside bottom, visibility=0
        ]

        for fmt in ['xyxy', 'xywh', 'cxcywh']:
            # corner case: empty list
            t = pxt.create_table('bbox_empty', {'bboxes': pxt.Json})
            validate_update_status(t.insert([{'bboxes': []}]), expected_rows=1)
            res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, width=640, height=480)).collect()
            assert res['out'][0] == []
            pxt.drop_table(t)

            cases: list[tuple[Any, dict[str, Any]]] = [(abs_boxes, {'width': 640, 'height': 480}), (rel_boxes, {})]
            for boxes, canvas_args in cases:
                input_bboxes = [convert_fmt(b[0], b[1], b[2], b[3], fmt) for b in boxes]
                t = pxt.create_table('bbox_clip', {'bboxes': pxt.Json})
                validate_update_status(t.insert([{'bboxes': input_bboxes}]), expected_rows=1)

                # basic clipping

                res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, **canvas_args)).collect()
                out = res['out'][0]
                assert len(out) == 5
                assert all(b is not None for b in out)

                # fully inside boxes unchanged
                assert out[0] == pytest.approx(input_bboxes[0])
                assert out[2] == pytest.approx(input_bboxes[2])

                # partially outside boxes have reduced w or h
                assert get_w(out[1], fmt) < get_w(input_bboxes[1], fmt)
                assert get_w(out[3], fmt) < get_w(input_bboxes[3], fmt)

                # fully outside box has zero height
                assert get_h(out[4], fmt) == pytest.approx(0, abs=1)

                # filtering min_visibility

                res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, **canvas_args, min_visibility=0.5)).collect()
                out = res['out'][0]
                for i in [0, 1, 2, 3]:
                    assert out[i] is not None
                assert out[4] is None

                res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, **canvas_args, min_visibility=0.7)).collect()
                out = res['out'][0]
                for i in [0, 2]:
                    assert out[i] is not None
                for i in [1, 3, 4]:
                    assert out[i] is None

                # filtering min_area

                if canvas_args:
                    # absolute: clipped areas [20000, 750, 2400, 1600, 0]
                    res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, **canvas_args, min_area=1000)).collect()
                    out = res['out'][0]
                    for i in [0, 2, 3]:
                        assert out[i] is not None
                    for i in [1, 4]:
                        assert out[i] is None
                else:
                    # relative: clipped areas [0.24, 0.0375, 0.04, 0.04, 0]
                    res = t.select(out=bboxes_clip_to_canvas(t.bboxes, fmt, **canvas_args, min_area=0.05)).collect()
                    out = res['out'][0]
                    assert out[0] is not None
                    for i in [1, 2, 3, 4]:
                        assert out[i] is None

                pxt.drop_table(t)

    def test_bboxes_clip_to_canvas_errors(self, uses_db: None) -> None:
        t = pxt.create_table('bbox_clip_err', {'bboxes': pxt.Json})

        # Missing both width/height for absolute coords
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='both width and height must be specified'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy')).collect()
        t.delete()

        # Missing height for absolute coords
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='both width and height must be specified'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640)).collect()
        t.delete()

        # Missing width for absolute coords
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='both width and height must be specified'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', height=480)).collect()
        t.delete()

        # width/height specified for relative coords
        t.insert([{'bboxes': [[0.1, 0.2, 0.3, 0.4]]}])
        with pytest.raises(pxt.Error, match='must not be specified for relative'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480)).collect()
        t.delete()

        # Invalid min_visibility
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='min_visibility must be between'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480, min_visibility=1.5)).collect()
        t.delete()

        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='min_visibility must be between'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480, min_visibility=-0.1)).collect()
        t.delete()

        # Negative min_area
        t.insert([{'bboxes': [[10, 20, 30, 40]]}])
        with pytest.raises(pxt.Error, match='min_area must be >= 0'):
            t.select(bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480, min_area=-1.0)).collect()
        t.delete()

        self._test_bbox_validation(t, bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480))

    def test_bboxes_clip_to_canvas_degenerate(self, uses_db: None) -> None:
        degenerate_boxes = [
            [10, 20, 10, 40],  # zero width (xyxy)
            [10, 20, 30, 20],  # zero height (xyxy)
            [10, 20, 10, 20],  # zero width and height (xyxy)
            [30, 40, 10, 20],  # negative width and height (xyxy, x2<x1, y2<y1)
        ]
        t = pxt.create_table('degenerate_clip', {'bboxes': pxt.Json})
        t.insert([{'bboxes': degenerate_boxes}])
        res = t.select(out=bboxes_clip_to_canvas(t.bboxes, 'xyxy', width=640, height=480)).collect()
        assert res['out'][0] == degenerate_boxes  # all passed through unchanged

    def _test_bbox_validation(self, t: pxt.Table, udf_call: Any) -> None:
        """Test that the bboxes parameter gets validated."""
        # Mixed int/float within a single box
        t.insert([{'bboxes': [[10, 20.0, 30, 40]]}])
        with pytest.raises(pxt.Error, match=r'either all int.*or all float'):
            t.select(udf_call).collect()
        t.delete()

        # Mixed absolute/relative across boxes
        t.insert([{'bboxes': [[10, 20, 30, 40], [0.1, 0.2, 0.3, 0.4]]}])
        with pytest.raises(pxt.Error, match=r'either all int.*or all float'):
            t.select(udf_call).collect()
        t.delete()

        # Wrong coordinate count
        t.insert([{'bboxes': [[10, 20, 30]]}])
        with pytest.raises(pxt.Error, match='exactly 4 coordinates'):
            t.select(udf_call).collect()
        t.delete()

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


# rudimentary bounding box utility functions for result validation


def convert_fmt(cx: float | int, cy: float | int, w: float | int, h: float | int, fmt: str) -> list:
    result: list
    if fmt == 'xyxy':
        result = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    elif fmt == 'xywh':
        result = [cx - w / 2, cy - h / 2, w, h]
    else:  # cxcywh
        result = [cx, cy, w, h]
    if cx < 1.0:
        # relative coords
        return result
    else:
        # absolute coords
        return [math.floor(x + 0.5) for x in result]


def get_w(box: list, fmt: str) -> int | float:
    result: float
    if fmt == 'xyxy':
        result = box[2] - box[0]
    elif fmt == 'xywh':
        result = box[2]
    else:  # cxcywh
        result = box[2]
    if box[0] < 1.0:
        return result
    else:
        return round(result)


def get_h(box: list, fmt: str) -> int | float:
    result: float
    if fmt == 'xyxy':
        result = box[3] - box[1]
    elif fmt == 'xywh':
        result = box[3]
    else:  # cxcywh
        result = box[3]
    if box[0] < 1.0:
        return result
    else:
        return round(result)


def get_aspect(box: list, fmt: str) -> float:
    return get_w(box, fmt) / get_h(box, fmt)


def crop_invariant(b_in: list, b_out: list, fmt: str) -> bool:
    """Crop: no dimension grows. Allows +1 tolerance for rounding."""
    return get_w(b_out, fmt) <= get_w(b_in, fmt) + 1 and get_h(b_out, fmt) <= get_h(b_in, fmt) + 1


def pad_invariant(b_in: list, b_out: list, fmt: str) -> bool:
    """Pad: no dimension shrinks. Allows -1 tolerance for rounding."""
    return get_w(b_out, fmt) >= get_w(b_in, fmt) - 1 and get_h(b_out, fmt) >= get_h(b_in, fmt) - 1

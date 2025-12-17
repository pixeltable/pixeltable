import pytest

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator

from ..utils import get_video_files, skip_test_if_not_installed


class TestVision:
    def test_eval(self, reset_db: None) -> None:
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
        from pixeltable.functions.vision import draw_bounding_boxes, eval_detections, mean_ap

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
            draw_bounding_boxes(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=True)
        ).collect()

    def test_draw_bounding_boxes(self, reset_db: None) -> None:
        skip_test_if_not_installed('yolox')
        from pixeltable.functions.yolox import yolox

        video_t = pxt.create_table('video_tbl', {'video': pxt.Video})
        # create frame view
        v = pxt.create_view('test_view', video_t, iterator=frame_iterator(video_t.video, fps=1))

        files = get_video_files()
        video_t.insert(video=files[-1])
        v.add_computed_column(frame_s=v.frame.resize([640, 480]))
        v.add_computed_column(detections_a=yolox(v.frame_s, model_id='yolox_nano'))

        from pixeltable.functions.vision import draw_bounding_boxes

        # default label colors
        _ = v.select(
            draw_bounding_boxes(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=True)
        ).collect()
        _ = v.select(
            draw_bounding_boxes(
                v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, fill=False, width=3
            )
        ).collect()
        _ = v.select(
            draw_bounding_boxes(v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, color='red')
        ).collect()

        # explicit box colors
        num_boxes = len(v.where(v.pos == 0).select(v.detections_a.bboxes).collect()[0, 0])
        box_colors = ['red'] * num_boxes
        _ = (
            v.where(v.pos == 0)
            .select(
                draw_bounding_boxes(
                    v.frame_s, boxes=v.detections_a.bboxes, labels=v.detections_a.labels, box_colors=box_colors
                )
            )
            .collect()
        )

        with pytest.raises(pxt.Error) as exc_info:
            # multiple color specifications
            _ = v.select(
                draw_bounding_boxes(
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
            _ = v.select(draw_bounding_boxes(v.frame_s, boxes=v.detections_a.bboxes, labels=[2])).collect()
        assert 'number of boxes and labels must match' in str(exc_info.value).lower()

        with pytest.raises(pxt.Error) as exc_info:
            # box_colors don't match boxes
            _ = v.select(draw_bounding_boxes(v.frame_s, boxes=v.detections_a.bboxes, box_colors=['red'])).collect()
        assert 'number of boxes and box colors must match' in str(exc_info.value).lower()

        # TODO: test font and font_size parameters in a system-independent way

    def test_draw_segmentation_masks(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import detr_for_segmentation
        from pixeltable.functions.vision import draw_segmentation_masks

        # Use a sample image URL
        sample_url = (
            'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/images/000000000034.jpg'
        )

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(
            seg=detr_for_segmentation(t.img, model_id='facebook/detr-resnet-50-panoptic', threshold=0.85)
        )
        t.insert(img=sample_url)

        # Test basic segmentation visualization
        result = t.select(draw_segmentation_masks(t.img, t.seg)).collect()
        assert len(result) == 1
        # Result should be a PIL Image
        assert result[0, 0] is not None

        # Test with different alpha values
        _ = t.select(draw_segmentation_masks(t.img, t.seg, alpha=0.3)).collect()
        _ = t.select(draw_segmentation_masks(t.img, t.seg, alpha=0.7, show_labels=False)).collect()

import pytest

import pixeltable as pxt
from pixeltable.functions.video import frame_iterator

from ..utils import (
    get_image_files,
    get_video_files,
    pxt_raises,
    rerun_on_network_error,
    skip_test_if_not_installed,
    validate_update_status,
)

pytestmark = pytest.mark.local('UDF/integration test')


@pytest.mark.very_expensive
@pytest.mark.xdist_group('yolox')
@rerun_on_network_error()
class TestYolox:
    def test_yolox(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')
        from pixeltable.functions.yolox import yolox

        t = pxt.create_table('yolox_test', {'image': pxt.Image | None})
        t.add_computed_column(detect_yolox_tiny=yolox(t.image, model_id='yolox_tiny'))
        t.add_computed_column(detect_yolox_nano=yolox(t.image, model_id='yolox_nano', threshold=0.2))
        t.add_computed_column(yolox_nano_bboxes=t.detect_yolox_nano.bboxes)
        images = get_image_files()[:10]
        validate_update_status(t.insert({'image': image} for image in images), expected_rows=10)
        rows = t.collect()
        # Verify correctly formed JSON
        assert all(list(result.keys()) == ['bboxes', 'labels', 'scores'] for result in rows['detect_yolox_tiny'])
        # Verify that bboxes are actually present in at least some of the rows.
        assert any(len(bboxes) > 0 for bboxes in rows['yolox_nano_bboxes'])

    @pytest.mark.local('exports a COCO dataset to the local filesystem')
    def test_yolox_coco_integration(self, uses_db: None) -> None:
        skip_test_if_not_installed('yolox')
        from pycocotools.coco import COCO

        from pixeltable.functions.yolox import yolo_to_coco, yolox

        base_t = pxt.create_table('videos', {'video': pxt.Video})
        view_t = pxt.create_view('frames', base_t, iterator=frame_iterator(base_t.video, fps=1))
        view_t.add_computed_column(detections=yolox(view_t.frame, model_id='yolox_m'))
        base_t.insert(video=get_video_files()[0])

        query = view_t.select({'image': view_t.frame, 'annotations': yolo_to_coco(view_t.detections)})
        path = query.to_coco_dataset()
        # we get a valid COCO dataset
        coco_ds = COCO(path)
        assert len(coco_ds.imgs) == view_t.count()

        # we call to_coco_dataset() again and get the cached dataset
        new_path = query.to_coco_dataset()
        assert path == new_path

        # the cache is invalidated when we add more data
        base_t.insert(video=get_video_files()[1])
        new_path = query.to_coco_dataset()
        assert path != new_path
        coco_ds = COCO(new_path)
        assert len(coco_ds.imgs) == view_t.count()

        # incorrect select list
        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION) as exc_info:
            _ = view_t.select({'image': view_t.frame, 'annotations': view_t.detections}).to_coco_dataset()
        assert '"annotations" is not a list' in str(exc_info.value)

        with pxt_raises(pxt.ErrorCode.MISSING_REQUIRED) as exc_info:
            _ = view_t.select(view_t.detections).to_coco_dataset()
        assert 'missing key "image"' in str(exc_info.value).lower()

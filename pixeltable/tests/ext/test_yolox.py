import pixeltable as pxt
from pixeltable.tests.utils import skip_test_if_not_installed, get_image_files, validate_update_status


class TestYolox:

    def test_yolox(self, test_client: pxt.Client):
        skip_test_if_not_installed('yolox')
        from pixeltable.ext.functions.yolox import yolox
        cl = test_client
        t = cl.create_table('yolox_test', {'image': pxt.ImageType()})
        t['detect_yolox_tiny'] = yolox(t.image, model_id='yolox_tiny')
        t['detect_yolox_nano'] = yolox(t.image, model_id='yolox_nano', threshold=0.2)
        t['yolox_nano_bboxes'] = t.detect_yolox_nano.bboxes
        images = get_image_files()[:10]
        validate_update_status(t.insert({'image': image} for image in images), expected_rows=10)
        rows = t.collect()
        # Verify correctly formed JSON
        assert all(list(result.keys()) == ['bboxes', 'labels', 'scores'] for result in rows['detect_yolox_tiny'])
        # Verify that bboxes are actually present in at least some of the rows.
        assert any(len(bboxes) > 0 for bboxes in rows['yolox_nano_bboxes'])

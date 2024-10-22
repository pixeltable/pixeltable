import pixeltable as pxt
from tests.utils import get_image_files, skip_test_if_not_installed


class TestFiftyone:
    def test_export_images(self, reset_db) -> None:
        skip_test_if_not_installed('fiftyone')
        import fiftyone as fo

        schema = {
            'id': pxt.Int,
            'image': pxt.Image,
            'classifications': pxt.Json,
            'detections': pxt.Json,
            'other_classifications': pxt.Json,
        }
        t = pxt.create_table('test_tbl', schema)
        images = get_image_files()[:10]
        t.insert({'id': n, 'image': images[n]} for n in range(len(images)))

        sample_cls = [{'label': 'cat', 'confidence': 0.5}, {'label': 'tiger', 'confidence': 0.3}]
        sample_det = [
            {'label': 'cat', 'bounding_box': [0.1, 0.2, 0.3, 0.4], 'confidence': 0.9},
            {'label': 'cat', 'bounding_box': [0.2, 0.2, 0.3, 0.4], 'confidence': 0.8}
        ]
        t.where(t.id < 5).update({'classifications': sample_cls})
        t.where(t.id < 3).update({'detections': sample_det})
        t.where(t.id < 2).update({'other_classifications': sample_cls})

        ds = pxt.io.export_images_as_fiftyone(
            t,
            t.image,
            classifications=t.classifications,
            detections=t.detections
        )
        assert len(ds) == 10
        assert ds.count('classifications') == 5
        assert ds.count('detections') == 3
        classifications = [c for c in ds.values('classifications') if c is not None]
        assert len(classifications) == 5
        for c in classifications:
            assert len(c.classifications) == 2
            assert c.classifications[0].label == 'cat'
            assert c.classifications[0].confidence == 0.5
            assert c.classifications[1].label == 'tiger'
            assert c.classifications[1].confidence == 0.3
        detections = [d for d in ds.values('detections') if d is not None]
        assert len(detections) == 3
        for d in detections:
            assert len(d.detections) == 2
            assert d.detections[0].label == 'cat'
            assert d.detections[0].confidence == 0.9
            assert d.detections[0].bounding_box == [0.1, 0.2, 0.3, 0.4]
            assert d.detections[1].label == 'cat'
            assert d.detections[1].confidence == 0.8
            assert d.detections[1].bounding_box == [0.2, 0.2, 0.3, 0.4]

        # Try a dynamically created image
        ds = pxt.io.export_images_as_fiftyone(
            t,
            t.image.rotate(180),
            classifications=t.classifications,
            detections=t.detections
        )
        assert len(ds) == 10

        # Multiple label sets
        ds = pxt.io.export_images_as_fiftyone(
            t,
            t.image,
            classifications={'first': t.classifications, 'other': t.other_classifications},
            detections=[t.detections]
        )
        assert len(ds) == 10
        assert ds.count('first') == 5
        assert ds.count('detections') == 3
        assert ds.count('other') == 2

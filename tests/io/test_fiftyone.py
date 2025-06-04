import sysconfig

import pytest
from PIL import Image
import numpy as np

import pixeltable as pxt
import pixeltable.exceptions as excs

from ..utils import get_image_files, skip_test_if_not_installed


@pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
class TestFiftyone:
    @pytest.mark.flaky(reruns=3)
    def test_export_images(self, reset_db: None) -> None:
        skip_test_if_not_installed('fiftyone')

        schema = {
            'id': pxt.Int,
            'image': pxt.Image,
            'classifications': pxt.Json,
            'detections': pxt.Json,
            'other_classifications': pxt.Json,
        }
        t = pxt.create_table('test_tbl', schema)
        images = get_image_files()[:5]
        images += [
            Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
            for _ in range(5)
        ]
        t.insert({'id': n, 'image': images[n]} for n in range(len(images)))

        sample_cls = [{'label': 'cat', 'confidence': 0.5}, {'label': 'tiger', 'confidence': 0.3}]
        sample_det = [
            {'label': 'cat', 'bounding_box': [0.1, 0.2, 0.3, 0.4], 'confidence': 0.9},
            {'label': 'cat', 'bounding_box': [0.2, 0.2, 0.3, 0.4], 'confidence': 0.8},
        ]
        t.where(t.id < 5).update({'classifications': sample_cls})
        t.where(t.id < 3).update({'detections': sample_det})
        t.where(t.id < 2).update({'other_classifications': sample_cls})

        ds = pxt.io.export_images_as_fo_dataset(t, t.image, classifications=t.classifications, detections=t.detections)
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
        ds = pxt.io.export_images_as_fo_dataset(
            t, t.image.rotate(180), classifications=t.classifications, detections=t.detections
        )
        assert len(ds) == 10

        # Multiple label sets
        ds = pxt.io.export_images_as_fo_dataset(
            t,
            t.image,
            classifications={'first': t.classifications, 'other': t.other_classifications},
            detections=[t.detections],
        )
        assert len(ds) == 10
        assert ds.count('first') == 5
        assert ds.count('detections') == 3
        assert ds.count('other') == 2

        for sample in ds:
            assert sample.filepath.endswith('.jpeg')

    def test_export_images_errors(self, reset_db: None) -> None:
        skip_test_if_not_installed('fiftyone')

        schema = {
            'id': pxt.Int,
            'image': pxt.Image,
            'classifications': pxt.Json,
            'detections': pxt.Json,
            'other_classifications': pxt.Json,
        }
        t = pxt.create_table('test_tbl', schema)
        img = get_image_files()[0]
        t.insert(id=0, image=img)

        with pytest.raises(excs.Error, match='`images` must be an expression of type Image'):
            pxt.io.export_images_as_fo_dataset(t, t.id)

        with pytest.raises(excs.Error, match='Invalid label name'):
            pxt.io.export_images_as_fo_dataset(t, t.image, classifications={'invalid name!@#': t.classifications})

        with pytest.raises(excs.Error, match='Duplicate label name'):
            pxt.io.export_images_as_fo_dataset(
                t, t.image, classifications={'labels': t.classifications}, detections={'labels': t.detections}
            )

        with pytest.raises(excs.Error, match='Invalid classifications data'):
            t.update({'classifications': {'a': 'b'}})
            pxt.io.export_images_as_fo_dataset(t, t.image, classifications=t.classifications)

        with pytest.raises(excs.Error, match='Invalid detections data'):
            t.update({'detections': {'a': 'b'}})
            pxt.io.export_images_as_fo_dataset(t, t.image, detections=t.detections)

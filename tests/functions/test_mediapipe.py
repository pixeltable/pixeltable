import pixeltable as pxt

from ..utils import get_image_files, skip_test_if_not_installed, validate_update_status


class TestMediapipe:
    def test_pose_landmarker(self, uses_db: None) -> None:
        skip_test_if_not_installed('mediapipe')
        from pixeltable.functions.mediapipe import pose_landmarker

        t = pxt.create_table('mediapipe_pose_test', {'image': pxt.Image})
        t.add_computed_column(pose=pose_landmarker(t.image, model='lite'))
        t.add_computed_column(detected=t.pose['detected'])
        images = get_image_files()[:5]
        validate_update_status(t.insert({'image': image} for image in images), expected_rows=5)
        rows = t.collect()
        assert all(isinstance(row['pose'], dict) for row in rows)
        assert all('detected' in row['pose'] and 'landmarks' in row['pose'] for row in rows)
        assert any(row['detected'] for row in rows)

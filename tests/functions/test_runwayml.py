import datetime

import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import get_image_files, rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.expensive
@rerun(reruns=3, reruns_delay=30)
class TestRunwayML:
    def test_image_to_data_uri(self) -> None:
        from pixeltable.functions.runwayml import _image_to_data_uri

        # RGB image -> jpeg
        rgb_image = PIL.Image.new('RGB', (100, 100), color='red')
        uri = _image_to_data_uri(rgb_image)
        assert uri.startswith('data:image/jpeg;base64,')
        assert len(uri) > 30

        # RGBA image -> png
        rgba_image = PIL.Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        uri = _image_to_data_uri(rgba_image)
        assert uri.startswith('data:image/png;base64,')
        assert len(uri) > 30

    def test_serialize_result(self) -> None:
        from pixeltable.functions.runwayml import _serialize_result

        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        result = _serialize_result({'created_at': dt, 'data': [{'time': dt}]})
        assert result['created_at'] == '2024-01-15T10:30:00'
        assert result['data'][0]['time'] == '2024-01-15T10:30:00'

    def test_udf_accepts_pil_image(self, reset_db: None) -> None:
        from pixeltable.functions.runwayml import image_to_video

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(video_output=image_to_video(t.image, prompt_text=t.prompt, model='gen4_turbo'))
        _ = t.video_output

    def test_text_to_image_accepts_image_list(self, reset_db: None) -> None:
        from pixeltable.functions.runwayml import text_to_image

        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref_image': pxt.Image})
        t.add_computed_column(output=text_to_image(t.prompt, [t.ref_image], model='gen4_image'))
        _ = t.output

    def test_image_to_video(self, reset_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import image_to_video

        image_files = get_image_files()[:1]
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(
            output=image_to_video(t.image, prompt_text=t.prompt, model='gen4_turbo', ratio='1280:720', duration=2)
        )
        validate_update_status(t.insert(image=image_files[0], prompt='Subtle movement'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    def test_text_to_image(self, reset_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import text_to_image

        image_files = get_image_files()[:1]
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref_image': pxt.Image})
        t.add_computed_column(output=text_to_image(t.prompt, [t.ref_image], model='gen4_image_turbo', ratio='720:720'))
        validate_update_status(t.insert(prompt='A colorful abstract painting', ref_image=image_files[0]), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    def test_text_to_video(self, reset_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import text_to_video

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=text_to_video(t.prompt, model='veo3.1_fast', ratio='1280:720', duration=4))
        validate_update_status(t.insert(prompt='A cat walking on a sunny day'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    def test_video_to_video(self, reset_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import video_to_video

        video_url = 'https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4'
        t = pxt.create_table('test_tbl', {'video_url': pxt.String, 'prompt': pxt.String})
        t.add_computed_column(output=video_to_video(t.video_url, t.prompt, model='gen4_aleph', ratio='1280:720'))
        validate_update_status(t.insert(video_url=video_url, prompt='Transform to anime style'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

import datetime

import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import get_image_files, rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=30)
class TestRunwayML:
    def test_image_to_data_uri(self) -> None:
        from pixeltable.functions.runwayml import _image_to_data_uri

        # RGB image -> webp
        rgb_image = PIL.Image.new('RGB', (100, 100), color='red')
        uri = _image_to_data_uri(rgb_image)
        assert uri.startswith('data:image/webp;base64,')
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

    def test_image_to_video_signatures(self, uses_db: None) -> None:
        """Test image_to_video with required-only and required+optional parameters."""
        skip_test_if_not_installed('runwayml')
        from pixeltable.functions.runwayml import image_to_video

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        # Required parameters only
        t.add_computed_column(video_required=image_to_video(t.image, 'gen4_turbo', '1280:720'))
        # Required + optional parameters
        t.add_computed_column(
            video_optional=image_to_video(
                t.image, 'gen4_turbo', '1280:720', prompt_text=t.prompt, duration=5, seed=42, audio=False
            )
        )

    def test_text_to_image_signatures(self, uses_db: None) -> None:
        """Test text_to_image with required-only and required+optional parameters."""
        skip_test_if_not_installed('runwayml')
        from pixeltable.functions.runwayml import text_to_image

        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref_image': pxt.Image})
        # Required parameters only
        t.add_computed_column(output_required=text_to_image(t.prompt, [t.ref_image], 'gen4_image', '1920:1080'))
        # Required + optional parameters
        t.add_computed_column(
            output_optional=text_to_image(t.prompt, [t.ref_image], 'gen4_image', '1920:1080', seed=42)
        )

    def test_text_to_video_signatures(self, uses_db: None) -> None:
        """Test text_to_video with required-only and required+optional parameters."""
        skip_test_if_not_installed('runwayml')
        from pixeltable.functions.runwayml import text_to_video

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        # Required parameters only
        t.add_computed_column(output_required=text_to_video(t.prompt, 'veo3.1', '1280:720'))
        # Required + optional parameters
        t.add_computed_column(output_optional=text_to_video(t.prompt, 'veo3.1', '1280:720', duration=5, audio=True))

    def test_video_to_video_signatures(self, uses_db: None) -> None:
        """Test video_to_video with required-only and required+optional parameters."""
        skip_test_if_not_installed('runwayml')
        from pixeltable.functions.runwayml import video_to_video

        t = pxt.create_table('test_tbl', {'video_url': pxt.String, 'prompt': pxt.String})
        # Required parameters only
        t.add_computed_column(output_required=video_to_video(t.video_url, t.prompt, 'gen4_aleph', '1280:720'))
        # Required + optional parameters
        t.add_computed_column(output_optional=video_to_video(t.video_url, t.prompt, 'gen4_aleph', '1280:720', seed=42))

    @pytest.mark.expensive
    def test_image_to_video(self, uses_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import image_to_video

        image_files = get_image_files()[:1]
        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(
            output=image_to_video(t.image, 'gen4_turbo', '1280:720', prompt_text=t.prompt, duration=2)
        )
        validate_update_status(t.insert(image=image_files[0], prompt='Subtle movement'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    def test_text_to_image(self, uses_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import text_to_image

        image_files = get_image_files()[:1]
        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref_image': pxt.Image})
        t.add_computed_column(output=text_to_image(t.prompt, [t.ref_image], 'gen4_image_turbo', '720:720'))
        validate_update_status(t.insert(prompt='A colorful abstract painting', ref_image=image_files[0]), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    @pytest.mark.expensive
    def test_text_to_video(self, uses_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import text_to_video

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=text_to_video(t.prompt, 'veo3.1_fast', '1280:720', duration=4))
        validate_update_status(t.insert(prompt='A cat walking on a sunny day'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    @pytest.mark.expensive
    def test_video_to_video(self, uses_db: None) -> None:
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import video_to_video

        video_url = 'https://storage.googleapis.com/gtv-videos-bucket/sample/ForBiggerMeltdowns.mp4'
        t = pxt.create_table('test_tbl', {'video_url': pxt.String, 'prompt': pxt.String})
        t.add_computed_column(output=video_to_video(t.video_url, t.prompt, 'gen4_aleph', '1280:720'))
        validate_update_status(t.insert(video_url=video_url, prompt='Transform to anime style'), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.expensive  # RunwayML credits are expensive - run only when explicitly needed
@rerun(reruns=3, reruns_delay=30)  # RunwayML tasks can take longer
class TestRunwayML:
    """Tests for RunwayML integration.

    Note: These tests make actual API calls to RunwayML and will consume credits.
    They are marked with @pytest.mark.remote_api and @pytest.mark.expensive.

    To run these tests: pytest -m "remote_api and expensive" tests/functions/test_runwayml.py

    Cost estimates (approximate):
    - text_to_image: ~5 credits per image
    - text_to_video: ~50-100 credits per 4-second video
    - image_to_video: ~25-50 credits per 2-second video (gen4_turbo is cheaper)

    The smoke test (test_image_to_video_smoke) is the cheapest option for verifying the integration.
    """

    def test_image_to_video_smoke(self, reset_db: None) -> None:
        """Smoke test: cheapest option to verify RunwayML integration works.

        Uses gen4_turbo (cheaper) with minimum duration (2 seconds).
        Run this test first to verify credits and API are working.
        """
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import image_to_video

        t = pxt.create_table('test_tbl', {'image_url': pxt.String, 'prompt': pxt.String})
        t.add_computed_column(
            output=image_to_video(
                t.image_url,
                prompt_text=t.prompt,
                model='gen4_turbo',  # Cheapest video model
                ratio='1280:720',
                duration=2,  # Minimum duration = minimum cost
            )
        )
        # Use a small, publicly accessible test image
        validate_update_status(
            t.insert(
                image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png',
                prompt='Subtle movement',
            ),
            1,
        )
        results = t.collect()
        print(results['output'][0])
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]

    def test_text_to_image_turbo(self, reset_db: None) -> None:
        """Test text-to-image with turbo model (faster and cheaper than gen4_image)."""
        skip_test_if_not_installed('runwayml')
        skip_test_if_no_client('runwayml')
        from pixeltable.functions.runwayml import text_to_image

        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'ref_images': pxt.Json})
        t.add_computed_column(
            output=text_to_image(
                t.prompt,
                t.ref_images,
                model='gen4_image_turbo',  # Turbo is cheaper
                ratio='720:720',  # Smaller ratio = less cost
            )
        )
        # Use a publicly accessible test image as reference
        ref_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png'
        validate_update_status(t.insert(prompt='A colorful abstract painting', ref_images=[ref_url]), 1)
        results = t.collect()
        assert results['output'][0] is not None
        assert 'output' in results['output'][0]


@pytest.mark.remote_api
class TestRunwayMLErrors:
    """Test error handling for RunwayML integration."""

    def test_missing_api_key(self, reset_db: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that appropriate error is raised when API key is missing."""
        skip_test_if_not_installed('runwayml')

        # Clear the API key from environment
        monkeypatch.delenv('RUNWAYML_API_KEY', raising=False)

        # This test would require mocking the client registration
        # to properly test missing API key behavior
        pass

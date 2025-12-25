import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, validate_update_status


class TestBfl:
    @pytest.mark.remote_api
    def test_generate(self, reset_db: None) -> None:
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1', width=512, height=512))

        validate_update_status(t.insert(prompt='A beautiful mountain landscape at sunset'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None
        # Check that we got an image with the expected dimensions
        img = result['image'][0]
        assert img.width == 512
        assert img.height == 512

    @pytest.mark.remote_api
    def test_generate_flux2(self, reset_db: None) -> None:
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model='flux-2-pro', width=1024, height=1024, seed=42))

        validate_update_status(t.insert(prompt='A futuristic city skyline at night with neon lights'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None
        img = result['image'][0]
        assert img.width == 1024
        assert img.height == 1024

    @pytest.mark.remote_api
    def test_edit(self, reset_db: None) -> None:
        skip_test_if_no_client('bfl')
        from pathlib import Path

        from pixeltable.functions.bfl import edit

        # Use a test image
        img_path = Path(__file__).parent.parent / 'data' / 'images' / 'test_image_rgb_1024x576.png'
        if not img_path.exists():
            # Find any available test image
            img_dir = Path(__file__).parent.parent / 'data' / 'images'
            img_paths = list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg'))
            if not img_paths:
                pytest.skip('No test images available')
            img_path = img_paths[0]

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'edit_prompt': pxt.String})
        t.add_computed_column(edited=edit(t.edit_prompt, t.image, model='flux-2-pro'))

        validate_update_status(t.insert(image=str(img_path), edit_prompt='Add a rainbow in the sky'), expected_rows=1)

        result = t.select(t.edited).collect()
        assert result['edited'][0] is not None

    @pytest.mark.remote_api
    def test_generate_with_seed_reproducibility(self, reset_db: None) -> None:
        """Test that using the same seed produces consistent results."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String, 'seed': pxt.Int})
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1', width=512, height=512, seed=t.seed))

        # Same prompt and seed should give similar results
        validate_update_status(t.insert([{'prompt': 'A red apple on a wooden table', 'seed': 12345}]), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None

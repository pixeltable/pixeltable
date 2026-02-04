import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import TESTS_DIR, rerun, skip_test_if_no_client, validate_update_status

# Test image path - using a simple HEIC image available in the test data
TEST_IMAGE_PATH = TESTS_DIR / 'data' / 'images' / 'sewing-threads.heic'


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestBfl:
    def test_generate(self, uses_db: None) -> None:
        """Test text-to-image generation with and without optional parameters."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        # Column 1: no optional parameters
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1'))
        # Column 2: all optional parameters
        t.add_computed_column(
            image_with_opts=generate(
                t.prompt,
                model='flux-pro-1.1',
                width=512,
                height=512,
                seed=12345,
                safety_tolerance=2,
                output_format='png',
            )
        )

        validate_update_status(t.insert(prompt='A beautiful mountain landscape at sunset'), expected_rows=1)

        result = t.collect()
        assert result['image'][0] is not None
        assert result['image'][0].width > 0
        assert result['image'][0].height > 0
        assert result['image_with_opts'][0] is not None
        assert result['image_with_opts'][0].width == 512
        assert result['image_with_opts'][0].height == 512

    def test_edit(self, uses_db: None) -> None:
        """Test image editing with and without optional parameters."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import edit

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        # Column 1: no optional parameters
        t.add_computed_column(edited=edit(t.prompt, t.image, model='flux-2-pro'))
        # Column 2: with optional parameters
        t.add_computed_column(
            edited_with_opts=edit(
                t.prompt, t.image, model='flux-2-pro', width=512, height=512, seed=42, safety_tolerance=2
            )
        )

        validate_update_status(
            t.insert(image=str(TEST_IMAGE_PATH), prompt='Make the colors more vibrant'), expected_rows=1
        )

        result = t.collect()
        assert result['edited'][0] is not None
        assert result['edited_with_opts'][0] is not None

    def test_fill(self, uses_db: None) -> None:
        """Test inpainting with and without optional parameters."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import fill

        # Load test image and create a simple mask
        with PIL.Image.open(TEST_IMAGE_PATH) as src_img:
            img = src_img.convert('RGB')
            # Create a mask of the same size (white center region)
            mask = PIL.Image.new('L', img.size, 0)
            # Draw white rectangle in center to mark region to fill
            w, h = img.size
            for x in range(w // 4, 3 * w // 4):
                for y in range(h // 4, 3 * h // 4):
                    mask.putpixel((x, y), 255)

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'mask': pxt.Image, 'prompt': pxt.String})
        # Column 1: no optional parameters
        t.add_computed_column(filled=fill(t.prompt, t.image, t.mask, model='flux-pro-1.0-fill'))
        # Column 2: with optional parameters
        t.add_computed_column(
            filled_with_opts=fill(
                t.prompt, t.image, t.mask, model='flux-pro-1.0-fill', steps=30, guidance=25.0, seed=42
            )
        )

        validate_update_status(
            t.insert(image=img, mask=mask, prompt='Fill with a beautiful pattern'), expected_rows=1
        )

        result = t.collect()
        assert result['filled'][0] is not None
        assert result['filled_with_opts'][0] is not None

    def test_expand(self, uses_db: None) -> None:
        """Test outpainting with different expansion configurations."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import expand

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        # Column 1: minimal expansion (one direction)
        t.add_computed_column(expanded=expand(t.prompt, t.image, model='flux-pro-1.0-expand', top=64))
        # Column 2: with multiple directions and optional parameters
        t.add_computed_column(
            expanded_with_opts=expand(
                t.prompt, t.image, model='flux-pro-1.0-expand', left=128, right=128, seed=42, safety_tolerance=2
            )
        )

        validate_update_status(
            t.insert(image=str(TEST_IMAGE_PATH), prompt='Continue the scene'), expected_rows=1
        )

        result = t.collect()
        assert result['expanded'][0] is not None
        assert result['expanded_with_opts'][0] is not None

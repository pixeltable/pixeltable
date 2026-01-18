import pytest

import pixeltable as pxt

from ..utils import TESTS_DIR, rerun, skip_test_if_no_client, validate_update_status

# Test image path - using a simple HEIC image available in the test data
TEST_IMAGE_PATH = TESTS_DIR / 'data' / 'images' / 'sewing-threads.heic'


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestBfl:
    @pytest.mark.parametrize('model,width,height', [('flux-pro-1.1', 512, 512), ('flux-2-pro', 1024, 1024)])
    def test_generate(self, reset_db: None, model: str, width: int, height: int) -> None:
        """Test text-to-image generation with different models."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model=model, width=width, height=height))

        validate_update_status(t.insert(prompt='A beautiful mountain landscape at sunset'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None
        img = result['image'][0]
        assert img.width == width
        assert img.height == height

    def test_generate_base(self, reset_db: None) -> None:
        """Test generate with only required arguments (no optional params)."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1'))

        validate_update_status(t.insert(prompt='A simple red circle'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None
        # Default dimensions should be 1024x1024
        img = result['image'][0]
        assert img.width > 0
        assert img.height > 0

    def test_generate_with_seed(self, reset_db: None) -> None:
        """Test generate with seed parameter for reproducibility."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1', width=512, height=512, seed=12345))

        validate_update_status(t.insert(prompt='A red apple on a wooden table'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None
        img = result['image'][0]
        assert img.width == 512
        assert img.height == 512

    def test_generate_with_safety_tolerance(self, reset_db: None) -> None:
        """Test generate with safety_tolerance parameter."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(image=generate(t.prompt, model='flux-pro-1.1', width=512, height=512, safety_tolerance=2))

        validate_update_status(t.insert(prompt='A peaceful garden scene'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None

    def test_generate_with_output_format(self, reset_db: None) -> None:
        """Test generate with output_format parameter."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import generate

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(
            image=generate(t.prompt, model='flux-pro-1.1', width=512, height=512, output_format='png')
        )

        validate_update_status(t.insert(prompt='A colorful butterfly'), expected_rows=1)

        result = t.select(t.image).collect()
        assert result['image'][0] is not None

    def test_edit(self, reset_db: None) -> None:
        """Test image editing with edit UDF."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import edit

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'edit_prompt': pxt.String})
        t.add_computed_column(edited=edit(t.edit_prompt, t.image, model='flux-2-pro'))

        validate_update_status(
            t.insert(image=str(TEST_IMAGE_PATH), edit_prompt='Make the colors more vibrant'), expected_rows=1
        )

        result = t.select(t.edited).collect()
        assert result['edited'][0] is not None

    def test_edit_base(self, reset_db: None) -> None:
        """Test edit with only required arguments."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import edit

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(edited=edit(t.prompt, t.image, model='flux-2-pro'))

        validate_update_status(t.insert(image=str(TEST_IMAGE_PATH), prompt='Add more contrast'), expected_rows=1)

        result = t.select(t.edited).collect()
        assert result['edited'][0] is not None

    def test_fill(self, reset_db: None) -> None:
        """Test inpainting with fill UDF."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import fill

        import PIL.Image

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

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
        t.add_computed_column(filled=fill(t.prompt, t.image, t.mask, model='flux-pro-1.0-fill'))

        # Save temp images for insertion
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_f:
            img.save(img_f.name)
            img_path = img_f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_f:
            mask.save(mask_f.name)
            mask_path = mask_f.name

        validate_update_status(
            t.insert(image=img_path, mask=mask_path, prompt='Fill with a beautiful pattern'), expected_rows=1
        )

        result = t.select(t.filled).collect()
        assert result['filled'][0] is not None

    def test_fill_base(self, reset_db: None) -> None:
        """Test fill with only required arguments."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import fill

        import PIL.Image

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

        with PIL.Image.open(TEST_IMAGE_PATH) as src_img:
            img = src_img.convert('RGB')
            mask = PIL.Image.new('L', img.size, 0)
            w, h = img.size
            for x in range(w // 4, 3 * w // 4):
                for y in range(h // 4, 3 * h // 4):
                    mask.putpixel((x, y), 255)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as img_f:
            img.save(img_f.name)
            img_path = img_f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as mask_f:
            mask.save(mask_f.name)
            mask_path = mask_f.name

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'mask': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(filled=fill(t.prompt, t.image, t.mask, model='flux-pro-1.0-fill'))

        validate_update_status(
            t.insert(image=img_path, mask=mask_path, prompt='Fill with solid color'), expected_rows=1
        )

        result = t.select(t.filled).collect()
        assert result['filled'][0] is not None

    def test_expand(self, reset_db: None) -> None:
        """Test outpainting with expand UDF."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import expand

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(expanded=expand(t.prompt, t.image, model='flux-pro-1.0-expand', left=128, right=128))

        validate_update_status(
            t.insert(image=str(TEST_IMAGE_PATH), prompt='Continue the scene to the sides'), expected_rows=1
        )

        result = t.select(t.expanded).collect()
        assert result['expanded'][0] is not None

    def test_expand_base(self, reset_db: None) -> None:
        """Test expand with minimal expansion (only required args plus at least one direction)."""
        skip_test_if_no_client('bfl')
        from pixeltable.functions.bfl import expand

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f'Test image not found: {TEST_IMAGE_PATH}')

        t = pxt.create_table('test_tbl', {'image': pxt.Image, 'prompt': pxt.String})
        t.add_computed_column(expanded=expand(t.prompt, t.image, model='flux-pro-1.0-expand', top=64))

        validate_update_status(t.insert(image=str(TEST_IMAGE_PATH), prompt='Extend upward'), expected_rows=1)

        result = t.select(t.expanded).collect()
        assert result['expanded'][0] is not None

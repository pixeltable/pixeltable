import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestFal:
    def test_text_to_image(self, uses_db: None) -> None:
        skip_test_if_not_installed('fal_client')
        skip_test_if_no_client('fal')
        from pixeltable.functions.fal import run

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=run(input={'prompt': t.prompt}, app='fal-ai/flux/schnell'))
        validate_update_status(t.insert(prompt='A friendly dinosaur playing tennis in a cornfield'), 1)
        results = t.collect()
        print(results['output'][0])
        assert 'images' in results['output'][0]
        assert len(results['output'][0]['images']) > 0

    def test_text_to_image_with_url(self, uses_db: None) -> None:
        skip_test_if_not_installed('fal_client')
        skip_test_if_no_client('fal')
        from pixeltable.functions.fal import run

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(response=run(input={'prompt': t.prompt}, app='fal-ai/flux/schnell'))
        t.add_computed_column(image=t.response['images'][0]['url'].astype(pxt.Image))
        validate_update_status(t.insert(prompt='A serene mountain landscape at sunset'), 1)
        results = t.collect()
        img = results['image'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size[0] > 0 and img.size[1] > 0

    def test_fast_sdxl(self, uses_db: None) -> None:
        skip_test_if_not_installed('fal_client')
        skip_test_if_no_client('fal')
        from pixeltable.functions.fal import run

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(
            response=run(
                input={'prompt': t.prompt, 'image_size': 'square_hd', 'num_inference_steps': 25}, app='fal-ai/fast-sdxl'
            )
        )
        t.add_computed_column(image=t.response['images'][0]['url'].astype(pxt.Image))
        validate_update_status(t.insert(prompt='A futuristic cityscape with flying cars'), 1)
        results = t.collect()
        img = results['image'][0]
        assert isinstance(img, PIL.Image.Image)
        # fast-sdxl with square_hd should produce a 1024x1024 image
        assert img.size == (1024, 1024)

    def test_multiple_images(self, uses_db: None) -> None:
        skip_test_if_not_installed('fal_client')
        skip_test_if_no_client('fal')
        from pixeltable.functions.fal import run

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(response=run(input={'prompt': t.prompt, 'num_images': 2}, app='fal-ai/flux/schnell'))
        validate_update_status(t.insert(prompt='An abstract painting with vibrant colors'), 1)
        results = t.collect()
        assert 'images' in results['response'][0]
        # Should return 2 images
        assert len(results['response'][0]['images']) == 2

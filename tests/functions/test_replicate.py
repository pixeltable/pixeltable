import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestReplicate:
    def test_chat_completions(self, reset_db) -> None:
        from pixeltable.functions.replicate import run

        skip_test_if_not_installed('replicate')
        skip_test_if_no_client('replicate')
        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t['output'] = run(
            input={'system_prompt': 'You are a helpful assistant.', 'prompt': t.input},
            ref='meta/meta-llama-3-8b-instruct'
        )
        validate_update_status(t.insert(input='What foods are rich in selenium?'), 1)
        results = t.collect()
        print(results['output'][0])
        assert len(results['output'][0]) > 0

    def test_image_generations(self, reset_db) -> None:
        from pixeltable.functions.replicate import run

        skip_test_if_not_installed('replicate')
        skip_test_if_no_client('replicate')
        t = pxt.create_table('test_tbl', {'prompt': pxt.String})

        t['response'] = run(
            input={'prompt': t.prompt, 'go_fast': True, 'megapixels': '1'},
            ref='black-forest-labs/flux-schnell'
        )
        t['image'] = t.response[0].astype(pxt.Image)
        validate_update_status(t.insert(
            prompt='Draw a pencil sketch of a friendly dinosaur playing tennis in a cornfield.'
        ), 1)
        results = t.collect()
        img = results['image'][0]
        assert isinstance(img, PIL.Image.Image)
        assert img.size == (1024, 1024)

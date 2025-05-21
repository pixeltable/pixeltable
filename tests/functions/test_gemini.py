from pathlib import Path
import pytest

import pixeltable as pxt

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
class TestGemini:
    def test_generate_content(self, reset_db: None) -> None:
        from pixeltable.functions.gemini import generate_content

        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')

        t = pxt.create_table('test_tbl', {'contents': pxt.String})
        t.add_computed_column(output=generate_content(t.contents, model='gemini-1.5-flash'))
        t.add_computed_column(
            output2=generate_content(
                t.contents,
                model='gemini-1.5-flash',
                candidate_count=3,
                stop_sequences=['\n'],
                max_output_tokens=300,
                temperature=1.0,
                top_p=0.95,
                top_k=40,
                response_mime_type='text/plain',
                presence_penalty=0.6,
                frequency_penalty=0.6,
            )
        )
        validate_update_status(t.insert(contents='Write a story about a magic backpack.'), expected_rows=1)
        results = t.collect()
        assert 'backpack' in results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        assert 'backpack' in results['output2'][0]['candidates'][0]['content']['parts'][0]['text']

    def test_generate_images(self, reset_db: None) -> None:
        from pixeltable.functions.gemini import generate_images
        from google.genai.types import GenerateImagesConfigDict

        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_images(t.prompt, model='imagen-3.0-generate-002'))
        config = GenerateImagesConfigDict(aspect_ratio='4:3')
        t.add_computed_column(output2=generate_images(t.prompt, model='imagen-3.0-generate-002', config=config))

        validate_update_status(t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1)
        results = t.collect()
        assert results['output'][0].size == (1024, 1024)
        assert results['output2'][0].size == (1280, 896)

    @pytest.mark.expensive
    def test_generate_videos(self, reset_db: None) -> None:
        from pixeltable.functions.gemini import generate_videos

        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_videos(t.prompt, model='veo-2.0-generate-001'))
        t.add_computed_column(metadata=t.output.get_metadata())
        validate_update_status(t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1)
        results = t.collect()
        print(results['output'][0])
        print(results['metadata'][0])
        assert Path(results['output'][0]).exists()
        assert results['metadata'][0]['streams'][0]['height'] == 720

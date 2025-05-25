from pathlib import Path

import pytest

import pixeltable as pxt

from ..conftest import DO_RERUN
from ..utils import skip_test_if_no_client, skip_test_if_not_installed, stock_price, validate_update_status


@pytest.mark.remote_api
@pytest.mark.flaky(reruns=3, reruns_delay=8, condition=DO_RERUN)
@pytest.mark.skip('temporarily disabled')
class TestGemini:
    def test_generate_content(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateContentConfigDict

        from pixeltable.functions.gemini import generate_content

        t = pxt.create_table('test_tbl', {'contents': pxt.String})
        t.add_computed_column(output=generate_content(t.contents, model='gemini-2.0-flash'))
        config = GenerateContentConfigDict(
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
        t.add_computed_column(output2=generate_content(t.contents, model='gemini-2.0-flash', config=config))
        validate_update_status(t.insert(contents='Write a story about a magic backpack.'), expected_rows=1)
        results = t.collect()
        text = results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        text2 = results['output2'][0]['candidates'][0]['content']['parts'][0]['text']
        print(text)
        print(text2)
        assert 'backpack' in text
        assert 'backpack' in text2

    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_content, invoke_tools

        tools = pxt.tools(stock_price)
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(response=generate_content(t.input, model='gemini-2.0-flash', tools=tools))
        t.insert(input='What is the stock price of NVDA today?')
        t.add_computed_column(tool_calls=invoke_tools(tools, t.response))

        results = t.collect()[0]
        assert results['tool_calls']['stock_price'] == [131.17]

    def test_generate_images(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateImagesConfigDict

        from pixeltable.functions.gemini import generate_images

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_images(t.prompt, model='imagen-3.0-generate-002'))
        config = GenerateImagesConfigDict(aspect_ratio='4:3')
        t.add_computed_column(output2=generate_images(t.prompt, model='imagen-3.0-generate-002', config=config))

        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )
        results = t.collect()
        assert results['output'][0].size == (1024, 1024)
        assert results['output2'][0].size == (1280, 896)

    @pytest.mark.expensive
    @pytest.mark.flaky(reruns=3, reruns_delay=30, condition=DO_RERUN)  # longer delay between reruns
    def test_generate_videos(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_videos(t.prompt, model='veo-2.0-generate-001'))
        t.add_computed_column(metadata=t.output.get_metadata())
        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )
        results = t.collect()
        print(results['output'][0])
        print(results['metadata'][0])
        assert Path(results['output'][0]).exists()
        assert results['metadata'][0]['streams'][0]['height'] == 720

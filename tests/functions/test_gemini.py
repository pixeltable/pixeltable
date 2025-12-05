from pathlib import Path

import pytest

import pixeltable as pxt
import pixeltable.functions as pxtf

from ..utils import (
    get_image_files,
    get_video_files,
    rerun,
    skip_test_if_no_client,
    skip_test_if_not_installed,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestGemini:
    def test_generate_content(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateContentConfigDict

        from pixeltable.functions.gemini import generate_content

        t = pxt.create_table('test_tbl', {'contents': pxt.String})
        t.add_computed_column(output=generate_content(t.contents, model='gemini-2.5-flash'))
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
        t.add_computed_column(output2=generate_content(t.contents, model='gemini-2.5-flash', config=config))
        validate_update_status(t.insert(contents='Write a story about a magic backpack.'), expected_rows=1)
        results = t.collect()
        text = results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        text2 = results['output2'][0]['candidates'][0]['content']['parts'][0]['text']
        print(text)
        print(text2)
        assert 'backpack' in text
        assert 'backpack' in text2

    def test_generate_content_multimodal(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_content

        images = get_image_files()[:2]

        t = pxt.create_table('test_tbl', {'id': pxt.Int, 'image': pxt.Image})
        t.add_computed_column(
            output=generate_content([t.image, "Describe what's in this image."], model='gemini-2.5-flash')
        )
        validate_update_status(t.insert({'id': n, 'image': image} for n, image in enumerate(images)), expected_rows=2)
        results = t.order_by(t.id).collect()
        assert 'French horn' in results['output'][0]['candidates'][0]['content']['parts'][0]['text']
        assert 'truck' in results['output'][1]['candidates'][0]['content']['parts'][0]['text']

    def test_tool_invocations(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions import gemini

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String}, if_exists='replace')
            t.add_computed_column(response=gemini.generate_content(t.prompt, model='gemini-2.0-flash', tools=tools))
            t.add_computed_column(tool_calls=gemini.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table)

    def test_generate_images(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from google.genai.types import GenerateImagesConfigDict

        from pixeltable.functions.gemini import generate_images

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(output=generate_images(t.prompt, model='imagen-4.0-generate-001'))
        config = GenerateImagesConfigDict(aspect_ratio='4:3')
        t.add_computed_column(output2=generate_images(t.prompt, model='imagen-4.0-generate-001', config=config))

        validate_update_status(
            t.insert(prompt='A giant pixel floating over the open ocean in a sea of data'), expected_rows=1
        )
        results = t.collect()
        assert results['output'][0].size == (1024, 1024)
        assert results['output2'][0].size == (1280, 896)

    # @pytest.mark.skip('Very expensive')
    # @pytest.mark.expensive
    @rerun(reruns=3, reruns_delay=30)  # longer delay between reruns
    def test_generate_videos(self, reset_db: None) -> None:
        skip_test_if_not_installed('google.genai')
        skip_test_if_no_client('gemini')
        from pixeltable.functions.gemini import generate_videos

        duration = 4
        t = pxt.create_table(
            'test_tbl', {'prompt': pxt.String, 'image': pxt.Image, 'video': pxt.Video, 'last_frame': pxt.Image}
        )
        t.add_computed_column(
            output=generate_videos(
                t.prompt, t.image, model='veo-3.1-generate-preview', last_frame=t.last_frame, config={'duration_seconds': duration}
            )
        )
        prompts = [
            # {
            #     'prompt': 'A giant pixel floating over the open ocean in a sea of data to the sound of ambient music'
            # },
            # {
            #     'prompt': 'Giraffes are foraging in a lush savannah as the leaves sway in the wind',
            #     'image': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000025.jpg',
            # },
            {
                'prompt': 'A taxi jumps into the air and comes to rest in its original position.',
                'image': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000001.jpg',
                'last_frame': 'https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/images/000000000001.jpg',
            }
        ]

        t.add_computed_column(metadata=t.output.get_metadata())
        validate_update_status(t.insert(prompts), expected_rows=len(prompts))

        results = t.collect()
        for i in range(len(results)):
            file_path = results['output'][i]
            print(f'Generated video: {file_path}')
            metadata = results['metadata'][i]
            print(f'Generated video metadata: {metadata}')
            assert Path(file_path).exists()

            # Validate metadata
            streams = metadata['streams']
            video_stream = next(s for s in streams if s['type'] == 'video')
            audio_stream = next(s for s in streams if s['type'] == 'audio')
            assert len(streams) == 2, metadata
            assert video_stream['height'] == 720, metadata
            assert video_stream['duration_seconds'] == duration, metadata
            assert audio_stream['duration_seconds'] == duration, metadata

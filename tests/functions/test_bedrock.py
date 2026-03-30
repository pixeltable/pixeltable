import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import (get_audio_files, get_image_files, get_video_files, rerun, skip_test_if_no_aws_credentials, skip_test_if_not_installed,
                     validate_update_status)
from .tool_utils import run_tool_invocations_test


#@pytest.mark.remote_api
#@rerun(reruns=3, reruns_delay=8)
class TestBedrock:
    def test_converse(self, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        system = [
            {
                'text': 'You are an app that creates playlists for a radio station that plays rock and pop music. '
                'Only return song names and the artist.'
            }
        ]

        t.add_computed_column(output=converse(messages, model_id='anthropic.claude-3-haiku-20240307-v1:0'))
        t.add_computed_column(
            output2=converse(
                messages,
                model_id='anthropic.claude-3-haiku-20240307-v1:0',
                system=system,
                inference_config={'temperature': 0.6},
                additional_model_request_fields={'top_k': 40},
            )
        )
        t.add_computed_column(response=t.output.output.message.content[0].text)
        t.add_computed_column(response2=t.output.output.message.content[0].text)

        t.insert(input='What were the 10 top charting pop singles in 2010?')
        results = t.collect()[0]
        assert 'Katy Perry' in results['response']
        assert 'Katy Perry' in results['response2']

    def test_invoke_model(self, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        body = {'inputText': t.text, 'dimensions': 256, 'normalize': True}
        t.add_computed_column(response=invoke_model(body, model_id='amazon.titan-embed-text-v2:0'))

        t.insert(text='Hello, world!')
        results = t.collect()[0]
        assert 'response' in results
        assert 'embedding' in results['response']
        assert len(results['response']['embedding']) == 256

    def test_invoke_model_twelvelabs(self, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        # image
        image_filepaths = get_image_files()[:2]
        t = pxt.create_table('image_tbl', {'image': pxt.Image})
        body = {
            'inputType': 'image',
            'image': {'mediaSource': {'base64String': t.image}},
        }
        t.add_computed_column(response=invoke_model(body, model_id='twelvelabs.marengo-embed-3-0-v1:0'))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        results = t.select(t.response).collect()
        assert 'response' in results[0]
        assert results[0]['response']['data'][0]['embedding'][0] is not None
        assert len(results[0]['response']['data'][0]['embedding']) == 512

        # audio
        audio_filepaths = get_audio_files()[:2]
        t = pxt.create_table('audio_tbl', {'audio': pxt.Audio})
        body = {
            'inputType': 'audio',
            'audio': {'mediaSource': {'base64String': t.audio}},
        }
        t.add_computed_column(response=invoke_model(body, model_id='twelvelabs.marengo-embed-3-0-v1:0'))
        validate_update_status(t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        results = t.select(t.response).collect()
        assert 'response' in results[0]
        assert results[0]['response']['data'][0]['embedding'][0] is not None
        assert len(results[0]['response']['data'][0]['embedding']) == 512

        # video
        video_filepaths = get_video_files()[:1]
        t = pxt.create_table('video_tbl', {'video': pxt.Video})
        body = {
            'inputType': 'video',
            'video': {'mediaSource': {'base64String': t.video}},
        }
        t.add_computed_column(response=invoke_model(body, model_id='twelvelabs.marengo-embed-3-0-v1:0'))
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        results = t.select(t.response).collect()
        assert 'response' in results[0]
        assert results[0]['response']['data'][0]['embedding'][0] is not None
        assert len(results[0]['response']['data'][0]['embedding']) == 512

        # text_image
        image_filepaths = get_image_files()[:2]
        t = pxt.create_table('text_image_tbl', {'image': pxt.Image})
        body = {
            'inputType': 'text_image',
            'text_image': {
                'inputText': 'man walking a dog',
                'mediaSource': {'base64String': t.image},
            },
        }
        t.add_computed_column(response=invoke_model(body, model_id='twelvelabs.marengo-embed-3-0-v1:0'))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        results = t.select(t.response).collect()
        assert 'response' in results[0]
        assert results[0]['response']['data'][0]['embedding'][0] is not None
        assert len(results[0]['response']['data'][0]['embedding']) == 512


    def test_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions import bedrock

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('test_tbl', {'prompt': pxt.String})
            messages = [{'role': 'user', 'content': [{'text': t.prompt}]}]
            t.add_computed_column(
                response=bedrock.converse(
                    messages, model_id='anthropic.claude-3-haiku-20240307-v1:0', tool_config=tools
                )
            )
            t.add_computed_column(tool_calls=bedrock.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_multiple_tool_use=False)

    @pytest.mark.parametrize(
        'model_id',
        [
            'amazon.titan-embed-text-v2:0',
            pytest.param(
                'amazon.nova-2-multimodal-embeddings-v1:0', marks=pytest.mark.skip(reason='Only available in us-east-1')
            ),
        ],
    )
    def test_embed_string(self, model_id: str, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('docs', {'text': pxt.String})
        t.add_embedding_index('text', string_embed=embed.using(model_id=model_id, dimensions=1024))

        t.insert(
            [
                {'text': 'Machine learning is a subset of artificial intelligence.'},
                {'text': 'Deep learning uses neural networks with many layers.'},
                {'text': 'Python is a popular programming language.'},
            ]
        )

        sim = t.text.similarity(string='What is machine learning?')
        results = t.order_by(sim, asc=False).limit(2).select(t.text, similarity=sim).collect()

        assert len(results) == 2
        # The ML-related text should be ranked first
        assert (
            'machine learning' in results['text'][0].lower() or 'artificial intelligence' in results['text'][0].lower()
        )

    @pytest.mark.parametrize(
        'model_id',
        [
            'amazon.titan-embed-image-v1',
            pytest.param(
                'amazon.nova-2-multimodal-embeddings-v1:0', marks=pytest.mark.skip(reason='Only available in us-east-1')
            ),
        ],
    )
    def test_embed_image(self, model_id: str, uses_db: None) -> None:
        skip_test_if_not_installed('boto3')
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('images', {'image': pxt.Image})
        t.add_embedding_index('image', image_embed=embed.using(model_id=model_id, dimensions=1024))
        img_paths = get_image_files()[:3]
        t.insert([{'image': p} for p in img_paths])

        sample_img = PIL.Image.open(img_paths[0])
        sim = t.image.similarity(image=sample_img)
        results = t.order_by(sim, asc=False).limit(2).select(t.image, similarity=sim).collect()

        assert len(results) == 2
        # The query image should be the most similar to itself
        assert results['similarity'][0] > results['similarity'][1]

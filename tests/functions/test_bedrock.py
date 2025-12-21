import PIL.Image
import pytest

import pixeltable as pxt

from ..utils import get_image_files, rerun, skip_test_if_no_aws_credentials, skip_test_if_not_installed
from .tool_utils import run_tool_invocations_test


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestBedrock:
    def test_converse(self, reset_db: None) -> None:
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

    def test_invoke_model(self, reset_db: None) -> None:
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

    def test_tool_invocations(self, reset_db: None) -> None:
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
    def test_embed_string(self, model_id: str, reset_db: None) -> None:
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
    def test_embed_image(self, model_id: str, reset_db: None) -> None:
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

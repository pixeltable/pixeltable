from collections.abc import Iterator
from pathlib import Path

import PIL.Image
import pytest

import pixeltable as pxt
from pixeltable import ResultSet
from pixeltable.config import Config

from ..utils import (
    get_audio_files,
    get_image_files,
    get_video_files,
    rerun,
    skip_test_if_no_aws_credentials,
    validate_update_status,
)
from .tool_utils import run_tool_invocations_test


@pytest.fixture()
def bedrock_us_east_1(uses_db: None) -> Iterator[str]:
    """Configure the Bedrock client for us-east-1. Yields the temp_location S3 URI."""
    Config.init(
        config_overrides={
            'bedrock.region_name': 'us-east-1',
            'bedrock.temp_location': 's3://pxt-test-us-east-1/bedrock_outputs/',
        },
        reinit=True,
    )
    yield 's3://pxt-test-us-east-1/bedrock_outputs/'


@pytest.fixture()
def bedrock_us_west_2(uses_db: None) -> Iterator[str]:
    """Configure the Bedrock client for us-west-2. Yields the temp_location S3 URI."""
    Config.init(
        config_overrides={
            'bedrock.region_name': 'us-west-2',
            'bedrock.temp_location': 's3://pxt-test-us-west-2/bedrock_outputs/',
        },
        reinit=True,
    )
    yield 's3://pxt-test-us-west-2/bedrock_outputs/'


def _tbl_name(model_id: str) -> str:
    """Sanitize a model ID into a valid Pixeltable table name."""
    return 'tbl_' + model_id.replace('.', '_').replace(':', '_').replace('-', '_')


def _run_converse_text(model_ids: list[str]) -> None:
    """Run a text converse call for each model_id and assert a text response."""
    from pixeltable.functions.bedrock import converse

    for model_id in model_ids:
        t = pxt.create_table(_tbl_name(model_id), {'input': pxt.String})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        t.add_computed_column(output=converse(messages, model_id=model_id))
        validate_update_status(t.insert(input='What is 2+2?'), expected_rows=1)
        results = t.collect()
        assert results[0]['output']['output']['message']['content']


def _assert_twelvelabs_embedding(results: ResultSet) -> None:
    assert 'response' in results[0]
    assert len(results[0]['response']['data'][0]['embedding']) == 512


def _assert_nova_text_response(results: ResultSet) -> None:
    assert results[0]['response']['output']['message']['content'][0]['text']


def _assert_openai_compat_response(results: ResultSet, model_id: str) -> None:
    assert results[0]['response']['choices'][0]['message']['content'], f'No response for {model_id}'


def _assert_text_similarity(t: pxt.Table) -> None:
    sim = t.text.similarity(string='What is machine learning?')
    results = t.order_by(sim, asc=False).limit(2).select(t.text, similarity=sim).collect()
    assert len(results) == 2
    assert 'machine learning' in results['text'][0].lower() or 'intelligence' in results['text'][0].lower()


def _assert_image_similarity(t: pxt.Table, img_paths: list) -> None:
    with PIL.Image.open(img_paths[0]) as sample_img:
        sim = t.image.similarity(image=sample_img)
        results = t.order_by(sim, asc=False).limit(2).select(t.image, similarity=sim).collect()
    assert len(results) == 2
    assert results['similarity'][0] > results['similarity'][1]


_TEXT_ROWS = [
    {'text': 'Machine learning is a subset of artificial intelligence.'},
    {'text': 'Deep learning uses neural networks with many layers.'},
    {'text': 'Python is a popular programming language.'},
]


def _image_url_body(t: pxt.Table) -> dict:
    """Request body for OpenAI-compatible image_url schema (Mistral, Gemma, NVIDIA, Moonshot, Qwen)."""
    return {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': t.image}},
                    {'type': 'text', 'text': 'Describe this image in one sentence.'},
                ],
            }
        ],
        'max_tokens': 256,
    }


def _converse_image_messages(t: pxt.Table) -> list:
    """Messages for the Converse unified image schema."""
    return [
        {
            'role': 'user',
            'content': [
                {'image': {'format': 'jpeg', 'source': {'bytes': t.image}}},
                {'text': 'What is in this image? Answer in one word.'},
            ],
        }
    ]


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestBedrock:
    def test_invoke_model_twelvelabs_marengo_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'image', 'image': {'mediaSource': {'base64String': t.image}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_text_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(
            response=invoke_model(
                {
                    'inputType': 'text_image',
                    'text_image': {'inputText': 'man walking a dog', 'mediaSource': {'base64String': t.image}},
                },
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_audio(self, uses_db: None, bedrock_us_east_1: str) -> None:
        # Audio auto-routes to StartAsyncInvoke.
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'audio': pxt.Audio})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'audio', 'audio': {'mediaSource': {'base64String': t.audio}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        audio_filepaths = get_audio_files(extension='.mp3')[:1]
        validate_update_status(t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_marengo_video(self, uses_db: None, bedrock_us_east_1: str) -> None:
        # Video auto-routes to StartAsyncInvoke.
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video})
        t.add_computed_column(
            response=invoke_model(
                {'inputType': 'video', 'video': {'mediaSource': {'base64String': t.video}}},
                model_id='twelvelabs.marengo-embed-3-0-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        _assert_twelvelabs_embedding(t.select(t.response).collect())

    def test_invoke_model_twelvelabs_pegasus_video(self, uses_db: None) -> None:
        """TwelveLabs Pegasus — invoke_model: video (sync)"""
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video})
        t.add_computed_column(
            response=invoke_model(
                # Pegasus uses a flat schema: inputPrompt + mediaSource at top level (no inputType nesting)
                {'inputPrompt': 'Describe this video.', 'mediaSource': {'base64String': t.video}},
                model_id='twelvelabs.pegasus-1-2-v1:0',
            )
        )

        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        results = t.select(t.response).collect()
        assert results[0]['response']['message']

    def test_invoke_model_anthropic_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(
            response=invoke_model(
                {
                    'anthropic_version': 'bedrock-2023-05-31',
                    'max_tokens': 256,
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {
                                    'type': 'image',
                                    'source': {'type': 'base64', 'media_type': 'image/jpeg', 'data': t.image},
                                },
                                {'type': 'text', 'text': 'Describe this image in one sentence.'},
                            ],
                        }
                    ],
                },
                model_id='anthropic.claude-3-haiku-20240307-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        results = t.select(t.response).collect()
        assert results[0]['response']['content'][0]['text']

    def test_converse_anthropic(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': [{'text': t.input}]}]
        t.add_computed_column(
            output=converse(
                messages,
                model_id='anthropic.claude-3-haiku-20240307-v1:0',
                system=[{'text': 'You are a helpful assistant. Keep answers short.'}],
                inference_config={'temperature': 0.6, 'maxTokens': 256},
                additional_model_request_fields={'top_k': 40},
            )
        )
        t.add_computed_column(response=t.output.output.message.content[0].text)
        validate_update_status(t.insert(input='What is the capital of France?'), expected_rows=1)
        assert 'Paris' in t.collect()[0]['response']

    def test_converse_tool_invocations(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions import bedrock

        def make_table(tools: pxt.Tools, tool_choice: pxt.ToolChoice) -> pxt.Table:
            t = pxt.create_table('tbl', {'prompt': pxt.String})
            messages = [{'role': 'user', 'content': [{'text': t.prompt}]}]
            t.add_computed_column(
                response=bedrock.converse(
                    messages, model_id='anthropic.claude-3-haiku-20240307-v1:0', tool_config=tools
                )
            )
            t.add_computed_column(tool_calls=bedrock.invoke_tools(tools, t.response))
            return t

        run_tool_invocations_test(make_table, test_multiple_tool_use=False)

    def test_invoke_model_nova_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(
            response=invoke_model(
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'image': {'format': 'jpeg', 'source': {'bytes': t.image}}},
                                {'text': 'Describe this image in one sentence.'},
                            ],
                        }
                    ],
                    'inferenceConfig': {'maxTokens': 256},
                },
                model_id='amazon.nova-lite-v1:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_nova_text_response(t.select(t.response).collect())

    def test_invoke_model_nova_video(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video})
        t.add_computed_column(
            response=invoke_model(
                {
                    'messages': [
                        {
                            'role': 'user',
                            'content': [
                                {'video': {'format': 'mp4', 'source': {'bytes': t.video}}},
                                {'text': 'Describe this video in one sentence.'},
                            ],
                        }
                    ],
                    'inferenceConfig': {'maxTokens': 256},
                },
                model_id='amazon.nova-lite-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        _assert_nova_text_response(t.select(t.response).collect())

    def test_converse_nova_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id='amazon.nova-lite-v1:0'))
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

    def test_converse_nova_video(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        t = pxt.create_table('tbl', {'video': pxt.Video})
        messages = [
            {
                'role': 'user',
                'content': [
                    {'video': {'format': 'mp4', 'source': {'bytes': t.video}}},
                    {'text': 'Describe this video in one sentence.'},
                ],
            }
        ]
        t.add_computed_column(output=converse(messages, model_id='amazon.nova-lite-v1:0'))
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

    def test_invoke_model_nova_reel(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'prompt': pxt.String})
        t.add_computed_column(
            video=invoke_model(
                {
                    'taskType': 'TEXT_VIDEO',
                    'textToVideoParams': {'text': t.prompt},
                    'videoGenerationConfig': {'durationSeconds': 6, 'fps': 24, 'dimension': '1280x720'},
                },
                model_id='amazon.nova-reel-v1:1',
            )
        )
        validate_update_status(t.insert(prompt='a dog running on a beach'), expected_rows=1)
        result = t.collect()[0]['video']
        assert result is not None and Path(result).exists()

    def test_invoke_model_titan_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'text': pxt.String})
        t.add_computed_column(
            response=invoke_model(
                {'inputText': t.text, 'dimensions': 256, 'normalize': True}, model_id='amazon.titan-embed-text-v2:0'
            )
        )
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['response']['embedding']) == 256

    def test_embed_titan_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        model_ids = ['amazon.titan-embed-text-v1', 'amazon.titan-embed-text-v2:0']
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'text': pxt.String})
            t.add_embedding_index('text', string_embed=embed.using(model_id=model_id))
            t.insert(_TEXT_ROWS)
            _assert_text_similarity(t)

    def test_embed_titan_text_custom_dimensions(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String})
        t.add_computed_column(embedding=embed(t.text, model_id='amazon.titan-embed-text-v2:0', dimensions=256))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 256

    def test_embed_titan_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        for model_id in ['amazon.titan-embed-image-v1']:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
            t.add_embedding_index('image', image_embed=embed.using(model_id=model_id))
            t.insert([{'image': p} for p in img_paths])
            _assert_image_similarity(t, img_paths)

    def test_embed_nova_multimodal_text(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String})
        t.add_embedding_index(
            'text', string_embed=embed.using(model_id='amazon.nova-2-multimodal-embeddings-v1:0', dimensions=1024)
        )
        t.insert(_TEXT_ROWS)
        _assert_text_similarity(t)

    def test_embed_nova_multimodal_image(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_embedding_index(
            'image', image_embed=embed.using(model_id='amazon.nova-2-multimodal-embeddings-v1:0', dimensions=1024)
        )
        t.insert([{'image': p} for p in img_paths])
        _assert_image_similarity(t, img_paths)

    def test_invoke_model_nova_multimodal_audio(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'audio': pxt.Audio})
        t.add_computed_column(
            response=invoke_model(
                {
                    'taskType': 'SINGLE_EMBEDDING',
                    'singleEmbeddingParams': {
                        'embeddingPurpose': 'GENERIC_INDEX',
                        'embeddingDimension': 1024,
                        'audio': {'format': 'mp3', 'source': {'bytes': t.audio}},
                    },
                },
                model_id='amazon.nova-2-multimodal-embeddings-v1:0',
            )
        )
        audio_filepaths = get_audio_files(extension='.mp3')[:1]
        validate_update_status(t.insert({'audio': p} for p in audio_filepaths), expected_rows=len(audio_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']

    def test_invoke_model_nova_multimodal_video(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'video': pxt.Video})
        t.add_computed_column(
            response=invoke_model(
                {
                    'taskType': 'SINGLE_EMBEDDING',
                    'singleEmbeddingParams': {
                        'embeddingPurpose': 'GENERIC_INDEX',
                        'embeddingDimension': 1024,
                        'video': {
                            'format': 'mp4',
                            'embeddingMode': 'AUDIO_VIDEO_COMBINED',
                            'source': {'bytes': t.video},
                        },
                    },
                },
                model_id='amazon.nova-2-multimodal-embeddings-v1:0',
            )
        )
        video_filepaths = get_video_files(extension='.mp4')[:1]
        validate_update_status(t.insert({'video': p} for p in video_filepaths), expected_rows=len(video_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']

    def test_converse_ai21(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['ai21.jamba-1-5-mini-v1:0', 'ai21.jamba-1-5-large-v1:0'])

    def test_embed_cohere_v3_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        for model_id in ['cohere.embed-english-v3', 'cohere.embed-multilingual-v3']:
            t = pxt.create_table(_tbl_name(model_id), {'text': pxt.String})
            t.add_embedding_index('text', string_embed=embed.using(model_id=model_id))
            t.insert(_TEXT_ROWS)
            _assert_text_similarity(t)

    def test_embed_cohere_v4_text(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String})
        t.add_computed_column(embedding=embed(t.text, model_id='cohere.embed-v4:0'))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 1536

    def test_embed_cohere_v4_text_custom_dimensions(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        t = pxt.create_table('tbl', {'text': pxt.String})
        t.add_computed_column(embedding=embed(t.text, model_id='cohere.embed-v4:0', dimensions=512))
        validate_update_status(t.insert(text='Hello, world!'), expected_rows=1)
        assert len(t.collect()[0]['embedding']) == 512

    def test_embed_cohere_v4_image(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import embed

        img_paths = get_image_files()[:3]
        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_embedding_index('image', image_embed=embed.using(model_id='cohere.embed-v4:0'))
        t.insert([{'image': p} for p in img_paths])
        _assert_image_similarity(t, img_paths)

    def test_invoke_model_cohere_v4_image(self, uses_db: None, bedrock_us_east_1: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'image': pxt.Image})
        t.add_computed_column(
            response=invoke_model(
                {
                    # Cohere embed-v4 uses inputs[] instead of messages[] for embedding requests.
                    'inputs': [{'content': [{'type': 'image_url', 'image_url': {'url': t.image}}]}],
                    'input_type': 'search_document',
                    'embedding_types': ['float'],
                },
                model_id='cohere.embed-v4:0',
            )
        )
        image_filepaths = get_image_files()[:1]
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.select(t.response).collect()[0]['response']['embeddings']['float'][0]

    def test_converse_deepseek(self, uses_db: None, bedrock_us_west_2: str) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(
            [
                'us.deepseek.r1-v1:0',  # cross-region inference profile
                'deepseek.v3.2',  # single-region bare ID
            ]
        )

    def test_converse_meta_llama_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'us.meta.llama4-maverick-17b-instruct-v1:0'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_mistral_vision_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        model_ids = ['us.mistral.pixtral-large-2502-v1:0', 'mistral.magistral-small-2509']
        image_filepaths = get_image_files()[:1]
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
            t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
            validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
            _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_mistral_vision_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'us.mistral.pixtral-large-2502-v1:0'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_converse_openai(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['openai.gpt-oss-safeguard-20b', 'openai.gpt-oss-safeguard-120b'])

    def test_invoke_model_gemma_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        model_ids = ['google.gemma-3-4b-it', 'google.gemma-3-12b-it', 'google.gemma-3-27b-it']
        image_filepaths = get_image_files()[:1]
        for model_id in model_ids:
            t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
            t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
            validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
            _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_gemma_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'google.gemma-3-12b-it'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_nvidia_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'nvidia.nemotron-nano-12b-v2'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_nvidia_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'nvidia.nemotron-nano-12b-v2'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_moonshot_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'moonshotai.kimi-k2.5'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_moonshot_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        model_id = 'moonshotai.kimi-k2.5'
        t = pxt.create_table(_tbl_name(model_id) + '_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text'], (
            f'No converse response for {model_id}'
        )

    def test_invoke_model_qwen_vl_image(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        image_filepaths = get_image_files()[:1]
        model_id = 'qwen.qwen3-vl-235b-a22b'
        t = pxt.create_table(_tbl_name(model_id), {'image': pxt.Image})
        t.add_computed_column(response=invoke_model(_image_url_body(t), model_id=model_id))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        _assert_openai_compat_response(t.select(t.response).collect(), model_id)

    def test_converse_qwen_image_and_text(self, uses_db: None) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import converse

        image_filepaths = get_image_files()[:1]
        t = pxt.create_table('tbl_qwen_vl_cv', {'image': pxt.Image})
        t.add_computed_column(output=converse(_converse_image_messages(t), model_id='qwen.qwen3-vl-235b-a22b'))
        validate_update_status(t.insert({'image': p} for p in image_filepaths), expected_rows=len(image_filepaths))
        assert t.collect()[0]['output']['output']['message']['content'][0]['text']

        _run_converse_text(['qwen.qwen3-next-80b-a3b'])

    def test_invoke_model_stability_image(self, uses_db: None, bedrock_us_west_2: str) -> None:
        skip_test_if_no_aws_credentials()
        from pixeltable.functions.bedrock import invoke_model

        t = pxt.create_table('tbl', {'prompt': pxt.String})
        t.add_computed_column(
            response=invoke_model(
                {'prompt': t.prompt, 'mode': 'text-to-image', 'aspect_ratio': '1:1', 'output_format': 'jpeg'},
                model_id='stability.sd3-5-large-v1:0',
            )
        )
        validate_update_status(t.insert(prompt='a cat sitting on a mat'), expected_rows=1)
        assert isinstance(t.collect()[0]['response']['images'][0], PIL.Image.Image)

    def test_converse_writer(self, uses_db: None, bedrock_us_west_2: str) -> None:
        skip_test_if_no_aws_credentials()
        _run_converse_text(['us.writer.palmyra-x4-v1:0', 'us.writer.palmyra-x5-v1:0'])

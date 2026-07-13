"""Tests for pixeltable.functions.databricks UDFs."""

from unittest import mock

import numpy as np
import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.local('UDF/integration test')


class TestDatabricksMocks:
    def test_chat_completions_mock(self, uses_db: None) -> None:
        mock_result = mock.Mock()
        mock_result.model_dump.return_value = {
            'choices': [{'message': {'content': 'hello', 'role': 'assistant'}}],
            'model': 'test-model',
        }

        mock_client = mock.AsyncMock()
        mock_client.chat.completions.create.return_value = mock_result

        with (
            mock.patch('pixeltable.functions.databricks.Env.require_package'),
            mock.patch('pixeltable.functions.databricks._databricks_client', return_value=mock_client),
        ):
            from pixeltable.functions.databricks import chat_completions

            t = pxt.create_table('test_dbx_chat', {'prompt': pxt.String})
            msgs = [{'role': 'user', 'content': t.prompt}]
            t.add_computed_column(out=chat_completions(msgs, model='test-endpoint'))
            validate_update_status(t.insert([{'prompt': 'hi'}]), 1)
            result = t.collect()[0]['out']
            assert result['choices'][0]['message']['content'] == 'hello'

    def test_embeddings_mock(self, uses_db: None) -> None:
        embedding = [0.1, 0.2, 0.3]
        mock_data = mock.Mock()
        mock_data.embedding = embedding
        mock_result = mock.Mock()
        mock_result.data = [mock_data]

        mock_client = mock.AsyncMock()
        mock_client.embeddings.create.return_value = mock_result

        with (
            mock.patch('pixeltable.functions.databricks.Env.require_package'),
            mock.patch('pixeltable.functions.databricks._databricks_client', return_value=mock_client),
        ):
            from pixeltable.functions.databricks import embeddings

            t = pxt.create_table('test_dbx_embed', {'text': pxt.String})
            t.add_computed_column(vec=embeddings(t.text, model='databricks-gte-large-en'))
            validate_update_status(t.insert([{'text': 'hello world'}]), 1)
            result = t.collect()[0]['vec']
            assert isinstance(result, np.ndarray)
            assert len(result) == 3


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestDatabricks:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('databricks')
        from pixeltable.functions.databricks import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=chat_completions(
                msgs,
                model='databricks-meta-llama-3-3-70b-instruct',
                model_kwargs={'max_tokens': 256, 'temperature': 0.2},
            )
        )
        validate_update_status(t.insert(input='Name three chemical elements discovered in the 21st century.'), 1)
        results = t.collect()
        assert len(results['output'][0]['choices'][0]['message']['content']) > 0

    def test_embeddings(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('databricks')
        from pixeltable.functions.databricks import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embed=embeddings(input=t.input, model='databricks-gte-large-en'))
        validate_update_status(t.insert(input='Databricks provides foundation model embeddings.'), 1)
        assert len(t.collect()['embed'][0]) > 0

    def test_embedding_index(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('databricks')
        from pixeltable.functions.databricks import embeddings

        t = pxt.create_table('docs', {'text': pxt.String})
        t.add_embedding_index('text', string_embed=embeddings.using(model='databricks-gte-large-en'))

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
        assert (
            'machine learning' in results['text'][0].lower() or 'artificial intelligence' in results['text'][0].lower()
        )

    @pytest.mark.skip(reason='Requires deployed supervisor/responses endpoint')
    def test_responses(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('databricks')
        from pixeltable.functions.databricks import responses

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(
            output=responses(input=[{'role': 'user', 'content': t.input}], model='databricks-claude-sonnet-4-5')
        )
        validate_update_status(t.insert(input='Summarize the benefits of declarative data pipelines.'), 1)
        assert len(t.collect()['output'][0]) > 0

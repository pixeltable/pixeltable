import pytest

import pixeltable as pxt
import pixeltable.type_system as ts

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.local('UDF/integration test')


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestNebius:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('nebius')
        from pixeltable.functions.nebius import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(
            chat_output=chat_completions(model='meta-llama/Llama-3.3-70B-Instruct', messages=t.input_msgs)
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()

    def test_embeddings(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('nebius')
        from pixeltable.functions.nebius import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embedding=embeddings(t.input, model='Qwen/Qwen3-Embedding-8B'))

        type_info = t._get_schema()
        assert isinstance(type_info['embedding'], ts.ArrayType)
        assert type_info['embedding'].shape == (4096,)

        validate_update_status(t.insert(input='Hello, world!'), 1)
        result = t.collect()
        assert len(result['embedding'][0]) == 4096

        # Via add_embedding_index(), which requires a statically known embedding dimension. The default
        # 4096 dimensions exceed pgvector's indexing limits, so request a truncated size instead.
        indexed_embedding = embeddings.using(model='Qwen/Qwen3-Embedding-8B', model_kwargs={'dimensions': 1024})
        t.add_embedding_index(t.input, embedding=indexed_embedding)
        validate_update_status(t.insert(input='Another sentence for you to index.'), 1)
        _ = t.head()

        sim = t.input.similarity(string='Indexing sentences is fun.')
        res = t.select(t.input, sim=sim).order_by(sim, asc=False).collect()
        assert res[0]['input'] == 'Another sentence for you to index.'

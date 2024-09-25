import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestMistral:
    def test_chat_completions(self, reset_db) -> None:
        from pixeltable.functions.mistralai import chat_completions

        skip_test_if_not_installed('mistralai')
        skip_test_if_no_client('mistral')
        t = pxt.create_table('test_tbl', {'input': pxt.StringType()})

        msgs = [{'role': 'user', 'content': t.input}]
        t['output'] = chat_completions(messages=msgs, model='mistral-small-latest')
        t['output2'] = chat_completions(
            messages=msgs,
            model='mistral-small-latest',
            temperature=0.8,
            top_p=0.95,
            max_tokens=300,
            min_tokens=100,
            stop=['\n'],
            random_seed=4171780,
            response_format={'type': 'text'},
            safe_prompt=True
        )
        validate_update_status(t.insert(input="What three species of fish have the highest mercury content?"), 1)
        results = t.collect()
        assert len(results['output'][0]['choices'][0]['message']['content']) > 0
        assert len(results['output2'][0]['choices'][0]['message']['content']) > 0

    def test_fim_completions(self, reset_db) -> None:
        from pixeltable.functions.mistralai import fim_completions

        skip_test_if_not_installed('mistralai')
        skip_test_if_no_client('mistral')
        t = pxt.create_table('test_tbl', {'input': pxt.StringType(), 'suffix': pxt.StringType(nullable=True)})

        t['output'] = fim_completions(prompt=t.input, model='codestral-latest')
        t['output2'] = fim_completions(
            prompt=t.input,
            model='codestral-latest',
            temperature=0.8,
            top_p=0.95,
            max_tokens=300,
            min_tokens=100,
            stop=['def'],
            random_seed=4171780,
            suffix=t.suffix
        )
        status = t.insert([
            {'input': 'def fibonacci(n: int):'},
            {'input': 'def fibonacci(n: int):', 'suffix': 'n = int(input("Enter a number: "))\nprint(fibonacci(n))'}
        ])
        validate_update_status(status, 2)
        results = t.collect()
        for out_col in ['output', 'output2']:
            for i in range(2):
                assert len(results[out_col][i]['choices'][0]['message']['content']) > 0

    def test_embeddings(self, reset_db) -> None:
        from pixeltable.functions.mistralai import embeddings

        skip_test_if_not_installed('mistralai')
        skip_test_if_no_client('mistral')
        t = pxt.create_table('test_tbl', {'input': pxt.StringType()})

        t.add_column(embed=embeddings(t.input, model='mistral-embed'))
        validate_update_status(t.insert(input='A chunk of text that will be embedded.'), 1)
        assert t.embed.col_type.shape == (1024,)
        assert len(t.collect()['embed'][0]) == 1024

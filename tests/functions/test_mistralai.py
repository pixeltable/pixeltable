import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestMistral:
    def test_mistral(self, reset_db) -> None:
        from pixeltable.functions.mistralai import completions

        skip_test_if_not_installed('mistralai')
        skip_test_if_no_client('mistral')
        t = pxt.create_table('test_tbl', {'input': pxt.StringType()})

        msgs = [{'role': 'user', 'content': t.input}]
        t['output'] = completions(messages=msgs, model='mistral-small-latest')
        t['output2'] = completions(
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

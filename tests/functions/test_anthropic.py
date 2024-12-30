import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestAnthropic:
    def test_anthropic(self, reset_db) -> None:
        from pixeltable.functions.anthropic import messages

        skip_test_if_not_installed('anthropic')
        skip_test_if_no_client('anthropic')
        t = pxt.create_table('test_tbl', {'input': pxt.String})

        msgs = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(output=messages(messages=msgs, model='claude-3-haiku-20240307'))
        t.add_computed_column(output2=messages(
            messages=msgs,
            model='claude-3-haiku-20240307',
            max_tokens=300,
            metadata={'user_id': 'pixeltable'},
            stop_sequences=['STOP'],
            system='You are an ordinary person walking down the street.',
            temperature=0.7,
            top_k=40,
            top_p=0.9,
        ))
        validate_update_status(t.insert(input="How's everything going today?"), 1)
        results = t.collect()
        assert len(results['output'][0]['content'][0]['text']) > 0
        assert len(results['output2'][0]['content'][0]['text']) > 0

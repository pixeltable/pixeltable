import pytest

import pixeltable as pxt

from ..utils import skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
#@pytest.mark.flaky(reruns=3, reruns_delay=8)
class TestDeepseek:
    def test_chat_completions(self, reset_db) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('deepseek')
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        from pixeltable.functions.deepseek import chat_completions

        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(model='deepseek-chat', messages=t.input_msgs))
        # test a bunch of the parameters
        t.add_computed_column(
            chat_output_2=chat_completions(
                model='deepseek-chat',
                messages=msgs,
                frequency_penalty=0.1,
                logprobs=True,
                top_logprobs=3,
                max_tokens=500,
                presence_penalty=0.1,
                stop=['\n'],
                temperature=0.7,
                top_p=0.8,
            )
        )
        t.add_computed_column(reasoning_output=chat_completions(model='deepseek-reasoner', messages=t.input_msgs))

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['chat_output_2'][0]['choices'][0]['message']['content'].lower()
        assert 'paris' in result['reasoning_output'][0]['choices'][0]['message']['content'].lower()

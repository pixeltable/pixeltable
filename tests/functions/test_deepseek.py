import pytest

import pixeltable as pxt

from ..utils import rerun_on_network_error, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='UDF/integration test')


@pytest.mark.remote_api
@pytest.mark.very_expensive
@rerun_on_network_error()
class TestDeepseek:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('deepseek')
        from pixeltable.functions.deepseek import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String | None})
        msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(model='deepseek-flash', messages=t.input_msgs))
        # test a bunch of the parameters
        t.add_computed_column(
            chat_output_2=chat_completions(
                model='deepseek-flash',
                messages=msgs,
                model_kwargs={
                    # thinking mode ignores temperature and the penalties
                    'extra_body': {'thinking': {'type': 'disabled'}},
                    'frequency_penalty': 0.1,
                    'logprobs': True,
                    'top_logprobs': 3,
                    'max_tokens': 500,
                    'presence_penalty': 0.1,
                    'stop': ['\n'],
                    'temperature': 0.7,
                    'top_p': 0.8,
                },
            )
        )

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        message = result['chat_output'][0]['choices'][0]['message']
        message_2 = result['chat_output_2'][0]['choices'][0]['message']
        assert 'paris' in message['content'].lower()
        assert len(message['reasoning_content']) > 0
        assert 'paris' in message_2['content'].lower()
        assert message_2.get('reasoning_content') is None

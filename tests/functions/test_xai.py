import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestXai:
    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [
            {'role': 'system', 'content': 'You are Grok, a helpful AI assistant.'},
            {'role': 'user', 'content': t.input},
        ]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat_completions(messages=t.input_msgs, model='grok-3'))

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['chat_output'][0]['choices'][0]['message']['content'].lower()

    def test_chat_completions_with_kwargs(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [
            {'role': 'system', 'content': 'You are Grok, a helpful AI assistant.'},
            {'role': 'user', 'content': t.input},
        ]
        t.add_computed_column(
            chat_output=chat_completions(
                messages=msgs,
                model='grok-3',
                model_kwargs={
                    'max_tokens': 100,
                    'temperature': 0.7,
                },
            )
        )

        validate_update_status(t.insert(input='Say hello in one word.'), 1)
        result = t.collect()
        assert result['chat_output'][0]['choices'][0]['message']['content'] is not None

    def test_image_generations(self, reset_db: None) -> None:
        skip_test_if_not_installed('openai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import image_generations

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        t.add_computed_column(generated_image=image_generations(t.prompt, model='grok-2-image'))

        validate_update_status(t.insert(prompt='A cute cartoon robot waving hello'), 1)
        result = t.collect()
        # Verify we got an image back
        assert result['generated_image'][0] is not None
        assert result['generated_image'][0].size[0] > 0
        assert result['generated_image'][0].size[1] > 0

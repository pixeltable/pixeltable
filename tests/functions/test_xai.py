import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestXai:
    def test_chat(self, reset_db: None) -> None:
        """Test the native xai-sdk chat UDF."""
        skip_test_if_not_installed('xai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import chat

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [
            {'role': 'system', 'content': 'You are Grok, a helpful AI assistant.'},
            {'role': 'user', 'content': t.input},
        ]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat(messages=t.input_msgs, model='grok-3'))
        t.add_computed_column(answer=t.chat_output['content'])

        validate_update_status(t.insert(input='What is the capital of France?'), 1)
        result = t.collect()
        assert 'paris' in result['answer'][0].lower()

    def test_chat_with_reasoning(self, reset_db: None) -> None:
        """Test the chat UDF with reasoning model grok-3-mini."""
        skip_test_if_not_installed('xai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import chat

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        msgs = [{'role': 'system', 'content': 'You are a math expert.'}, {'role': 'user', 'content': t.input}]
        t.add_computed_column(input_msgs=msgs)
        t.add_computed_column(chat_output=chat(messages=t.input_msgs, model='grok-3-mini', reasoning_effort='high'))

        validate_update_status(t.insert(input='What is 101 * 3?'), 1)
        result = t.collect()
        assert '303' in result['chat_output'][0]['content']
        # Check that reasoning tokens were used
        if 'usage' in result['chat_output'][0]:
            assert result['chat_output'][0]['usage'].get('reasoning_tokens', 0) > 0

    def test_chat_completions(self, reset_db: None) -> None:
        """Test the OpenAI-compatible chat_completions UDF."""
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
        """Test chat_completions with model_kwargs."""
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
                messages=msgs, model='grok-3', model_kwargs={'max_tokens': 100, 'temperature': 0.7}
            )
        )

        validate_update_status(t.insert(input='Say hello in one word.'), 1)
        result = t.collect()
        assert result['chat_output'][0]['choices'][0]['message']['content'] is not None

    def test_vision(self, reset_db: None) -> None:
        """Test vision/image understanding with Grok."""
        skip_test_if_not_installed('xai')
        skip_test_if_no_client('xai')
        from pixeltable.functions.xai import vision

        t = pxt.create_table('test_tbl', {'image_url': pxt.String, 'question': pxt.String})
        t.add_computed_column(analysis=vision(prompt=t.question, image_url=t.image_url, model='grok-4', detail='high'))
        t.add_computed_column(answer=t.analysis['content'])

        # Use a simple public image
        validate_update_status(
            t.insert(
                image_url='https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/1200px-Cat_August_2010-4.jpg',
                question='What animal is in this image?',
            ),
            1,
        )
        result = t.collect()
        assert 'cat' in result['answer'][0].lower()

    def test_image_generations(self, reset_db: None) -> None:
        """Test image generation with Grok."""
        skip_test_if_not_installed('xai')
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

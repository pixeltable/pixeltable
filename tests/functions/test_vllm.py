import pytest

import pixeltable as pxt

from ..utils import skip_test_if_not_installed, validate_update_status


@pytest.mark.expensive
class TestVllm:
    def test_chat_completions(self, uses_db: None) -> None:
        skip_test_if_not_installed('vllm')
        from pixeltable.functions import vllm

        t = pxt.create_table('test_tbl', {'input': pxt.String})

        t.add_computed_column(
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': t.input},
            ]
        )

        # We use a small model for testing.
        t.add_computed_column(
            output=vllm.chat_completions(
                t.messages, model='Qwen/Qwen2.5-0.5B-Instruct', model_kwargs={'max_model_len': 512}
            )
        )
        t.add_computed_column(
            output2=vllm.chat_completions(
                t.messages,
                model='Qwen/Qwen2.5-0.5B-Instruct',
                model_kwargs={'max_model_len': 512},
                sampling_kwargs={'max_tokens': 100, 'temperature': 0.7, 'top_p': 0.9, 'top_k': 20},
            )
        )

        validate_update_status(t.insert(input='What are some edible species of fish?'), expected_rows=1)
        result = t.collect()['output'][0]
        result2 = t.collect()['output2'][0]
        print(result)
        assert len(result['choices'][0]['message']['content']) > 0
        assert len(result2['choices'][0]['message']['content']) > 0

        vllm.cleanup()  # Clean up the model cache after the test

    def test_generate(self, uses_db: None) -> None:
        skip_test_if_not_installed('vllm')
        from pixeltable.functions import vllm

        t = pxt.create_table('test_gen_tbl', {'prompt': pxt.String})

        t.add_computed_column(
            output=vllm.generate(t.prompt, model='Qwen/Qwen2.5-0.5B-Instruct', model_kwargs={'max_model_len': 512})
        )

        validate_update_status(t.insert(prompt='The capital of France is'), expected_rows=1)
        result = t.collect()['output'][0]
        print(result)
        assert len(result['choices'][0]['text']) > 0

        vllm.cleanup()  # Clean up the model cache after the test

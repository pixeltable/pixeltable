import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs

from ..utils import skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
#@pytest.mark.flaky(reruns=3, reruns_delay=8)
class TestTogether:
    def test_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        from pixeltable.functions.together import completions

        t.add_computed_column(
            output=completions(prompt=t.input, model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', stop=['\n'])
        )
        t.add_computed_column(
            output_2=completions(
                prompt=t.input,
                model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                max_tokens=300,
                stop=['\n'],
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                logprobs=1,
                echo=True,
                n=3,
                # The safety model sometimes triggers even on an innocuous prompt, causing an
                # exception to be thrown. Unclear if there's a reliable way to test this param.
                # safety_model='Meta-Llama/Meta-Llama-Guard-3-8B'
            )
        )
        validate_update_status(t.insert(input='I am going to the '), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['text']) > 0
        assert len(result['output_2'][0]['choices'][0]['text']) > 0

    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': t.input}]
        from pixeltable.functions.together import chat_completions

        t.add_computed_column(
            output=chat_completions(messages=messages, model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', stop=['\n'])
        )
        t.add_computed_column(
            output_2=chat_completions(
                messages=messages,
                model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                max_tokens=300,
                stop=['\n'],
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                logprobs=1,
                # echo=True,
                n=3,
                safety_model='Meta-Llama/Meta-Llama-Guard-3-8B',
                response_format={'type': 'json_object'},
            )
        )
        validate_update_status(t.insert(input='Give me a typical example of a JSON structure.'), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['message']) > 0
        assert len(result['output_2'][0]['choices'][0]['message']) > 0

    def test_embeddings(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        t = pxt.create_table('test_tbl', {'input': pxt.String})
        from pixeltable.functions.together import embeddings

        t.add_computed_column(embed=embeddings(input=t.input, model='togethercomputer/m2-bert-80M-8k-retrieval'))
        validate_update_status(t.insert(input='Together AI provides a variety of embeddings models.'), 1)
        assert len(t.collect()['embed'][0]) > 0

    @pytest.mark.expensive
    def test_image_generations(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        t = pxt.create_table('test_tbl', {'input': pxt.String, 'negative_prompt': pxt.String})
        from pixeltable.functions.together import image_generations

        t.add_computed_column(img=image_generations(t.input, model='stabilityai/stable-diffusion-xl-base-1.0'))
        t.add_computed_column(
            img_2=image_generations(
                t.input,
                model='stabilityai/stable-diffusion-xl-base-1.0',
                steps=30,
                seed=4171780,
                negative_prompt=t.negative_prompt,
            )
        )
        validate_update_status(
            t.insert(
                [
                    {'input': 'A friendly dinosaur playing tennis in a cornfield'},
                    {'input': 'A friendly dinosaur playing tennis in a cornfield', 'negative_prompt': 'tennis court'},
                ]
            ),
            2,
        )
        assert t.collect()['img'][0].size == (1024, 1024)
        assert t.collect()['img_2'][0].size == (1024, 1024)
        assert t.collect()['img'][1].size == (1024, 1024)
        assert t.collect()['img_2'][1].size == (1024, 1024)

    # This ensures that the test will be skipped, rather than returning an error, when no API key is
    # available (for example, when a PR runs in CI).
    @staticmethod
    def skip_test_if_no_together_client() -> None:
        try:
            import pixeltable.functions.together

            _ = pixeltable.functions.together._together_client()
        except excs.Error as exc:
            pytest.skip(str(exc))

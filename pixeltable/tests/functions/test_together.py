import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.tests.utils import skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
class TestTogether:

    def test_completions(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': pxt.StringType()})
        from pixeltable.functions.together import completions
        t.add_column(output=completions(prompt=t.input, model='mistralai/Mixtral-8x7B-v0.1', stop=['\n']))
        t.add_column(output_2=completions(
            prompt=t.input,
            model='mistralai/Mixtral-8x7B-v0.1',
            max_tokens=300,
            stop=['\n'],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            logprobs=1,
            echo=True,
            n=3,
            safety_model='Meta-Llama/Llama-Guard-7b'
        ))
        validate_update_status(t.insert(input='I am going to the '), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['text']) > 0
        assert len(result['output_2'][0]['choices'][0]['text']) > 0

    def test_chat_completions(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': pxt.StringType()})
        messages = [{'role': 'user', 'content': t.input}]
        from pixeltable.functions.together import chat_completions
        t.add_column(output=chat_completions(messages=messages, model='mistralai/Mixtral-8x7B-v0.1', stop=['\n']))
        t.add_column(output_2=chat_completions(
            messages=messages,
            model='mistralai/Mixtral-8x7B-Instruct-v0.1',
            max_tokens=300,
            stop=['\n'],
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            logprobs=1,
            echo=True,
            n=3,
            safety_model='Meta-Llama/Llama-Guard-7b',
            response_format={'type': 'json_object'}
        ))
        validate_update_status(t.insert(input='Give me a typical example of a JSON structure.'), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['message']) > 0
        assert len(result['output_2'][0]['choices'][0]['message']) > 0

    def test_embeddings(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': pxt.StringType()})
        from pixeltable.functions.together import embeddings
        t.add_column(embed=embeddings(input=t.input, model='togethercomputer/m2-bert-80M-8k-retrieval'))
        validate_update_status(t.insert(input='Together AI provides a variety of embeddings models.'), 1)
        assert len(t.collect()['embed'][0]) > 0

    def test_image_generations(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('together')
        TestTogether.skip_test_if_no_together_client()
        cl = test_client
        t = cl.create_table(
            'test_tbl',
            {'input': pxt.StringType(), 'negative_prompt': pxt.StringType(nullable=True)}
        )
        from pixeltable.functions.together import image_generations
        t.add_column(img=image_generations(t.input, model='runwayml/stable-diffusion-v1-5'))
        t.add_column(img_2=image_generations(
            t.input,
            model='stabilityai/stable-diffusion-2-1',
            steps=30,
            seed=4178780,
            height=768,
            width=512,
            negative_prompt=t.negative_prompt
        ))
        validate_update_status(t.insert([
            {'input': 'A friendly dinosaur playing tennis in a cornfield'},
            {'input': 'A friendly dinosaur playing tennis in a cornfield',
             'negative_prompt': 'tennis court'}
        ]), 2)
        assert t.collect()['img'][0].size == (512, 512)
        assert t.collect()['img_2'][0].size == (512, 768)
        assert t.collect()['img'][1].size == (512, 512)
        assert t.collect()['img_2'][1].size == (512, 768)

    # This ensures that the test will be skipped, rather than returning an error, when no API key is
    # available (for example, when a PR runs in CI).
    @staticmethod
    def skip_test_if_no_together_client() -> None:
        try:
            import pixeltable.functions.together
            _ = pixeltable.functions.together.together_client()
        except excs.Error as exc:
            pytest.skip(str(exc))

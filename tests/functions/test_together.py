import pytest

import pixeltable as pxt

from ..utils import rerun, skip_test_if_no_client, skip_test_if_not_installed, validate_update_status


@pytest.mark.remote_api
@rerun(reruns=3, reruns_delay=8)
class TestTogether:
    def test_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        skip_test_if_no_client('together')
        from pixeltable.functions.together import completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(
            output=completions(
                prompt=t.input, model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', model_kwargs={'stop': ['\n']}
            )
        )
        t.add_computed_column(
            output_2=completions(
                prompt=t.input,
                model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                model_kwargs={
                    'max_tokens': 300,
                    'stop': ['\n'],
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repetition_penalty': 1.1,
                    'logprobs': 1,
                    'echo': True,
                    'n': 3,
                },
            )
        )
        validate_update_status(t.insert(input='I am going to the '), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['text']) > 0
        assert len(result['output_2'][0]['choices'][0]['text']) > 0

    def test_chat_completions(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        skip_test_if_no_client('together')
        from pixeltable.functions.together import chat_completions

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        messages = [{'role': 'user', 'content': t.input}]
        t.add_computed_column(
            output=chat_completions(
                messages=messages, model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', model_kwargs={'stop': ['\n']}
            )
        )
        t.add_computed_column(
            output_2=chat_completions(
                messages=messages,
                model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                model_kwargs={
                    'max_tokens': 300,
                    'stop': ['\n'],
                    'temperature': 0.7,
                    'top_p': 0.9,
                    'top_k': 40,
                    'repetition_penalty': 1.1,
                    'logprobs': 1,
                    'n': 3,
                    'safety_model': 'Meta-Llama/Meta-Llama-Guard-3-8B',
                    'response_format': {'type': 'json_object'},
                },
            )
        )
        validate_update_status(t.insert(input='Give me a typical example of a JSON structure.'), 1)
        result = t.collect()
        assert len(result['output'][0]['choices'][0]['message']) > 0
        assert len(result['output_2'][0]['choices'][0]['message']) > 0

    def test_embeddings(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        skip_test_if_no_client('together')
        from pixeltable.functions.together import embeddings

        t = pxt.create_table('test_tbl', {'input': pxt.String})
        t.add_computed_column(embed=embeddings(input=t.input, model='BAAI/bge-base-en-v1.5'))
        validate_update_status(t.insert(input='Together AI provides a variety of embeddings models.'), 1)
        assert len(t.collect()['embed'][0]) > 0

    @pytest.mark.expensive
    def test_image_generations(self, reset_db: None) -> None:
        skip_test_if_not_installed('together')
        skip_test_if_no_client('together')
        from pixeltable.functions.together import image_generations

        t = pxt.create_table('test_tbl', {'input': pxt.String, 'negative_prompt': pxt.String})
        t.add_computed_column(
            img=image_generations(t.input, model='black-forest-labs/FLUX.1-schnell', model_kwargs={'steps': 5})
        )
        t.add_computed_column(
            img_2=image_generations(
                t.input,
                model='black-forest-labs/FLUX.1-schnell',
                model_kwargs={
                    'steps': 5,
                    'width': 768,
                    'height': 1024,
                    'seed': 4171780,
                    'negative_prompt': t.negative_prompt,
                },
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
        assert t.collect()['img_2'][0].size == (768, 1024)
        assert t.collect()['img'][1].size == (1024, 1024)
        assert t.collect()['img_2'][1].size == (768, 1024)

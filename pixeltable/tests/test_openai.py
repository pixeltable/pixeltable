from typing import cast

import PIL.Image
import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.env import Env
from pixeltable.tests.utils import SAMPLE_IMAGE_URL, skip_test_if_not_installed
from pixeltable.type_system import StringType, ImageType


class TestOpenai:

    def test_chat_completions(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import chat_completions
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": t.input}
        ]
        t.add_column(input_msgs=msgs)
        t.add_column(chat_output=chat_completions(model='gpt-3.5-turbo', messages=t.input_msgs))
        # with inlined messages
        t.add_column(chat_output_2=chat_completions(model='gpt-3.5-turbo', messages=msgs))
        # test a bunch of the parameters
        t.add_column(chat_output_3=chat_completions(
            model='gpt-3.5-turbo', messages=msgs, frequency_penalty=0.1, logprobs=True, top_logprobs=3,
            max_tokens=500, n=3, presence_penalty=0.1, seed=4171780, stop=['\n'], temperature=0.7, top_p=0.8,
            user='pixeltable'
        ))
        # test with JSON output enforced
        t.add_column(chat_output_4=chat_completions(
            model='gpt-3.5-turbo', messages=msgs, response_format={'type': 'json_object'}
        ))
        # TODO Also test the `tools` and `tool_choice` parameters.
        t.insert(input='Give me an example of a typical JSON structure.')
        result = t.collect()
        assert len(result['chat_output'][0]) > 0
        assert len(result['chat_output_2'][0]) > 0
        assert len(result['chat_output_3'][0]) > 0
        assert len(result['chat_output_4'][0]) > 0

        # TODO This should probably not be throwing an exception, but rather logging the error in
        # the Pixeltable virtual columns for error records.
        with pytest.raises(excs.ExprEvalError) as exc_info:
            t.insert(input='Say something interesting.')
        assert "\\'messages\\' must contain the word \\'json\\'" in str(exc_info.value)

    def test_gpt_4_vision(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'prompt': StringType(), 'img': ImageType()})
        from pixeltable.functions.openai import chat_completions, vision
        from pixeltable.functions.string import str_format
        t.add_column(response=vision(prompt="What's in this image?", image=t.img))
        # Also get the response the low-level way, by calling chat_completions
        msgs = [
            {'role': 'user',
             'content': [
                 {'type': 'text', 'text': t.prompt},
                 {'type': 'image_url', 'image_url': {
                     'url': str_format('data:image/png;base64,{0}', t.img.b64_encode())
                 }}
             ]}
        ]
        t.add_column(response_2=chat_completions(model='gpt-4-vision-preview', messages=msgs, max_tokens=300).choices[0].message.content)
        t.insert(prompt="What's in this image?", img=SAMPLE_IMAGE_URL)
        result = t.collect()['response_2'][0]
        assert len(result) > 0

    def test_embeddings(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        from pixeltable.functions.openai import embeddings
        t = cl.create_table('test_tbl', {'input': StringType()})
        t.add_column(ada_embed=embeddings(model='text-embedding-ada-002', input=t.input))
        t.add_column(text_3=embeddings(model='text-embedding-3-small', input=t.input, user='pixeltable'))
        t.insert(input='Say something interesting.')
        _ = t.head()

    def test_moderations(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import moderations
        t.add_column(moderation=moderations(input=t.input))
        t.add_column(moderation_2=moderations(input=t.input, model='text-moderation-stable'))
        t.insert(input='Say something interesting.')
        _ = t.head()

    def test_image_generations(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import image_generations
        t.add_column(img=image_generations(t.input))
        # Test dall-e-2 options
        t.add_column(img_2=image_generations(
            t.input, model='dall-e-2', n=3, size='512x512', user='pixeltable'
        ))
        # Test dall-e-3 options
        t.add_column(img_3=image_generations(
            t.input, model='dall-e-3', quality='hd', size='1792x1024', style='natural', user='pixeltable'
        ))
        t.insert(input='A friendly dinosaur playing tennis in a cornfield')
        assert t.collect()['img'][0].size == (1024, 1024)
        assert t.collect()['img_2'][0].size == (512, 512)
        assert t.collect()['img_3'][0].size == (1792, 1024)

    @staticmethod
    def skip_test_if_no_openai_client() -> None:
        try:
            _ = Env.get().openai_client
        except excs.Error as exc:
            pytest.skip(str(exc))

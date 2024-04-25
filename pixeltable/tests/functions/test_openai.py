import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.tests.utils import SAMPLE_IMAGE_URL, skip_test_if_not_installed, validate_update_status
from pixeltable.type_system import StringType, ImageType


@pytest.mark.remote_api
class TestOpenai:

    def test_audio(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import speech, transcriptions, translations
        t.add_column(speech=speech(t.input, model='tts-1', voice='onyx'))
        t.add_column(speech_2=speech(t.input, model='tts-1', voice='onyx', response_format='flac', speed=1.05))
        t.add_column(transcription=transcriptions(t.speech, model='whisper-1'))
        t.add_column(transcription_2=transcriptions(
            t.speech, model='whisper-1', language='en', prompt='Transcribe the contents of this recording.'
        ))
        t.add_column(translation=translations(t.speech, model='whisper-1'))
        t.add_column(translation_2=translations(
            t.speech, model='whisper-1', prompt='Translate the recording from Spanish into English.', temperature=0.05
        ))
        validate_update_status(t.insert([
            {'input': 'I am a banana.'},
            {'input': 'Es fácil traducir del español al inglés.'}
        ]), expected_rows=2)
        # The audio generation -> transcription loop on these examples should be simple and clear enough
        # that the unit test can reliably expect the output closely enough to pass these checks.
        results = t.collect()
        assert results[0]['transcription']['text'] in ['I am a banana.', "I'm a banana."]
        assert results[0]['transcription_2']['text'] in ['I am a banana.', "I'm a banana."]
        assert 'easy to translate' in results[1]['translation']['text']
        assert 'easy to translate' in results[1]['translation_2']['text']

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
        validate_update_status(t.insert(input='Give me an example of a typical JSON structure.'), 1)
        result = t.collect()
        assert len(result['chat_output'][0]['choices'][0]['message']['content']) > 0
        assert len(result['chat_output_2'][0]['choices'][0]['message']['content']) > 0
        assert len(result['chat_output_3'][0]['choices'][0]['message']['content']) > 0
        assert len(result['chat_output_4'][0]['choices'][0]['message']['content']) > 0

        # When OpenAI gets a request with `response_format` equal to `json_object`, but the prompt does not
        # contain the string "json", it refuses the request.
        # TODO This should probably not be throwing an exception, but rather logging the error in
        # `t.chat_output_4.errormsg` etc.
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
        validate_update_status(t.insert(prompt="What's in this image?", img=SAMPLE_IMAGE_URL), 1)
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
        validate_update_status(t.insert(input='Say something interesting.'), 1)
        _ = t.head()

    def test_moderations(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import moderations
        t.add_column(moderation=moderations(input=t.input))
        t.add_column(moderation_2=moderations(input=t.input, model='text-moderation-stable'))
        validate_update_status(t.insert(input='Say something interesting.'), 1)
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
            t.input, model='dall-e-2', size='512x512', user='pixeltable'
        ))
        validate_update_status(t.insert(input='A friendly dinosaur playing tennis in a cornfield'), 1)
        assert t.collect()['img'][0].size == (1024, 1024)
        assert t.collect()['img_2'][0].size == (512, 512)

    @pytest.mark.skip('Test is expensive and slow')
    def test_image_generations_dall_e_3(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestOpenai.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import image_generations
        # Test dall-e-3 options
        t.add_column(img_3=image_generations(
            t.input, model='dall-e-3', quality='hd', size='1792x1024', style='natural', user='pixeltable'
        ))
        validate_update_status(t.insert(input='A friendly dinosaur playing tennis in a cornfield'), 1)
        assert t.collect()['img_3'][0].size == (1792, 1024)

    # This ensures that the test will be skipped, rather than returning an error, when no API key is
    # available (for example, when a PR runs in CI).
    @staticmethod
    def skip_test_if_no_openai_client() -> None:
        try:
            import pixeltable.functions.openai
            _ = pixeltable.functions.openai.openai_client()
        except excs.Error as exc:
            pytest.skip(str(exc))

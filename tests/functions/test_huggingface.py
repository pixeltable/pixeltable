import sys
import sysconfig
from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.type_system as ts

from ..conftest import DO_RERUN
from ..utils import (
    SAMPLE_IMAGE_URL,
    ReloadTester,
    get_audio_files,
    get_image_files,
    get_sentences,
    reload_catalog,
    skip_test_if_not_installed,
    validate_update_status,
)


@pytest.mark.flaky(reruns=3, reruns_delay=15, condition=DO_RERUN)  # Guard against connection errors downloading models
class TestHuggingface:
    def test_hf_function(self, reset_db: None) -> None:
        skip_test_if_not_installed('sentence_transformers')
        from pixeltable.functions.huggingface import sentence_transformer

        t = pxt.create_table('test_tbl', {'input': pxt.String, 'bool_col': pxt.Bool})
        model_id = 'intfloat/e5-large-v2'
        t.add_computed_column(e5=sentence_transformer(t.input, model_id=model_id))
        sents = get_sentences()
        status = t.insert({'input': s, 'bool_col': True} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # verify handling of constant params
        with pytest.raises(ValueError) as exc_info:
            t.add_computed_column(e5_2=sentence_transformer(t.input, model_id=t.input))
        assert ': parameter model_id must be a constant value' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            t.add_computed_column(
                e5_2=sentence_transformer(t.input, model_id=model_id, normalize_embeddings=t.bool_col)
            )
        assert ': parameter normalize_embeddings must be a constant value' in str(exc_info.value)

        # make sure this doesn't cause an exception
        # TODO: is there some way to capture the output?
        t.describe()

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_sentence_transformer(self, reset_db: None, reload_tester: ReloadTester) -> None:
        skip_test_if_not_installed('sentence_transformers')
        from pixeltable.functions.huggingface import sentence_transformer, sentence_transformer_list

        t = pxt.create_table('test_tbl', {'input': pxt.String, 'input_list': pxt.Json})
        sents = get_sentences(10)
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        model_ids = ['sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-reranker-base']
        num_dims = [768, 768]
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed{idx}'
            t.add_computed_column(
                **{col_name: sentence_transformer(t.input, model_id=model_id, normalize_embeddings=True)}
            )
            assert t._get_schema()[col_name].is_array_type()
            list_col_name = f'embed_list{idx}'
            t.add_computed_column(
                **{list_col_name: sentence_transformer_list(t.input_list, model_id=model_id, normalize_embeddings=True)}
            )
            assert t._get_schema()[list_col_name] == ts.JsonType(nullable=True)

        def verify_row(row: dict[str, Any]) -> None:
            for idx, (_, d) in enumerate(zip(model_ids, num_dims)):
                assert row[f'embed{idx}'].shape == (d,)
                assert len(row[f'embed_list{idx}']) == len(sents)
                assert all(len(v) == d for v in row[f'embed_list{idx}'])

        verify_row(t.tail(1)[0])

        # execution still works after reload
        _ = reload_tester.run_query(t.select())
        reload_tester.run_reload_test()

        t = pxt.get_table('test_tbl')
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_cross_encoder(self, reset_db: None) -> None:
        skip_test_if_not_installed('sentence_transformers')
        from pixeltable.functions.huggingface import cross_encoder, cross_encoder_list

        t = pxt.create_table('test_tbl', {'input': pxt.String, 'input_list': pxt.Json})
        sents = get_sentences(10)
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        model_ids = ['cross-encoder/ms-marco-MiniLM-L-6-v2', 'cross-encoder/ms-marco-TinyBERT-L-2-v2']
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed{idx}'
            t.add_computed_column(**{col_name: cross_encoder(t.input, t.input, model_id=model_id)})
            assert t._get_schema()[col_name] == ts.FloatType(nullable=True)
            list_col_name = f'embed_list{idx}'
            t.add_computed_column(**{list_col_name: cross_encoder_list(t.input, t.input_list, model_id=model_id)})
            assert t._get_schema()[list_col_name] == ts.JsonType(nullable=True)

        def verify_row(row: dict[str, Any]) -> None:
            for idx in range(len(model_ids)):
                assert len(row[f'embed_list{idx}']) == len(sents)
                assert all(isinstance(v, float) for v in row[f'embed_list{idx}'])

        verify_row(t.tail(1)[0])

        # execution still works after reload
        reload_catalog()
        t = pxt.get_table('test_tbl')
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    def test_clip(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import clip

        t = pxt.create_table('test_tbl', {'text': pxt.String, 'img': pxt.Image})
        num_rows = 10
        sents = get_sentences(num_rows)
        imgs = get_image_files()[:num_rows]
        status = t.insert({'text': text, 'img': img} for text, img in zip(sents, imgs))
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        model_ids = ['openai/clip-vit-base-patch32', 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K']
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed_text{idx}'
            t.add_computed_column(**{col_name: clip(t.text, model_id=model_id)})
            assert t._get_schema()[col_name].is_array_type()
            col_name = f'embed_img{idx}'
            t.add_computed_column(**{col_name: clip(t.img, model_id=model_id)})
            assert t._get_schema()[col_name].is_array_type()

        def verify_row(row: dict[str, Any]) -> None:
            for idx, _ in enumerate(model_ids):
                assert row[f'embed_text{idx}'].shape == (512,)
                assert row[f'embed_img{idx}'].shape == (512,)

        verify_row(t.tail(1)[0])

        # execution still works after reload
        reload_catalog()
        t = pxt.get_table('test_tbl')
        status = t.insert({'text': text, 'img': img} for text, img in zip(sents, imgs))
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_detr_for_object_detection(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import detr_for_object_detection
        from pixeltable.utils import coco

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(
            detect=detr_for_object_detection(t.img, model_id='facebook/detr-resnet-50', threshold=0.8)
        )
        status = t.insert(img=SAMPLE_IMAGE_URL)
        assert status.num_rows == 1
        assert status.num_excs == 0
        result = t.select(t.detect).collect()[0]['detect']
        assert 'orange' in result['label_text']
        assert 'bowl' in result['label_text']
        assert 'broccoli' in result['label_text']
        label_text = {coco.COCO_2017_CATEGORIES[i] for i in result['labels']}
        assert 'orange' in label_text
        assert 'bowl' in label_text
        assert 'broccoli' in label_text

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_vit_for_image_classification(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import vit_for_image_classification

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t.add_computed_column(
            img_class=vit_for_image_classification(t.img, model_id='google/vit-base-patch16-224', top_k=3)
        )
        validate_update_status(t.insert(img=SAMPLE_IMAGE_URL), expected_rows=1)
        result = t.select(t.img_class).collect()[0]['img_class']
        assert result['labels'] == [962, 935, 937]
        assert result['label_text'] == ['meat loaf, meatloaf', 'mashed potato', 'broccoli']

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    @pytest.mark.skipif(sys.version_info >= (3, 13), reason='Not working on Python 3.13+')
    def test_speech2text_for_conditional_generation(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import speech2text_for_conditional_generation

        t = pxt.create_table('test_tbl', {'audio': pxt.Audio})
        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t.add_computed_column(
            transcription=speech2text_for_conditional_generation(t.audio, model_id='facebook/s2t-small-librispeech-asr')
        )
        t.add_computed_column(
            translation=speech2text_for_conditional_generation(
                t.audio, model_id='facebook/s2t-medium-mustc-multilingual-st', language='fr'
            )
        )

        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        result = t.collect()
        assert 'administration' in result['transcription'][0]
        assert 'construire' in result['translation'][0]

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_text_generation(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import text_generation

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        test_prompts = ['The weather today is', 'Machine learning is']

        # Test with GPT-2 (lightweight model)
        t.add_computed_column(completion=text_generation(t.prompt, model_id='gpt2', max_length=20, temperature=0.5))

        validate_update_status(t.insert({'prompt': p} for p in test_prompts), expected_rows=2)
        results = t.select(t.completion).collect()

        # Verify we got text completions
        assert len(results) == 2
        for result in results:
            assert isinstance(result['completion'], str)
            assert len(result['completion'].strip()) > 0

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_text_classification(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import text_classification

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        test_texts = ['I love this product!', 'This is terrible.']

        # Test with a sentiment analysis model
        t.add_computed_column(
            sentiment=text_classification(t.text, model_id='cardiffnlp/twitter-roberta-base-sentiment-latest', top_k=2)
        )

        validate_update_status(t.insert({'text': text} for text in test_texts), expected_rows=2)
        results = t.select(t.sentiment).collect()

        # Verify we got classification results
        assert len(results) == 2
        for result in results:
            assert isinstance(result['sentiment'], list)
            assert len(result['sentiment']) <= 2  # top_k=2
            for item in result['sentiment']:
                assert 'label' in item
                assert 'score' in item

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_image_captioning(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import image_captioning

        t = pxt.create_table('test_tbl', {'img': pxt.Image})

        # Test with BLIP model
        t.add_computed_column(
            caption=image_captioning(t.img, model_id='Salesforce/blip-image-captioning-base', max_length=30)
        )

        validate_update_status(t.insert(img=SAMPLE_IMAGE_URL), expected_rows=1)
        result = t.select(t.caption).collect()[0]

        # Verify we got a caption
        assert isinstance(result['caption'], str)
        assert len(result['caption'].strip()) > 0

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_text_summarization(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import text_summarization

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        long_text = (
            'Machine learning is a method of data analysis that automates analytical model building. '
            'It is a branch of artificial intelligence based on the idea that systems can learn from data, '
            'identify patterns and make decisions with minimal human intervention.'
        )

        # Test with BART model
        t.add_computed_column(
            summary=text_summarization(t.text, model_id='facebook/bart-large-cnn', max_length=50, min_length=10)
        )

        validate_update_status(t.insert(text=long_text), expected_rows=1)
        result = t.select(t.summary).collect()[0]

        # Verify we got a summary
        assert isinstance(result['summary'], str)
        assert len(result['summary'].strip()) > 0
        assert len(result['summary']) < len(long_text)  # Should be shorter than original

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_question_answering(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import question_answering

        t = pxt.create_table('test_tbl', {'context': pxt.String, 'question': pxt.String})
        context = 'Paris is the capital of France. It is known for the Eiffel Tower.'
        question = 'What is the capital of France?'

        # Test with DistilBERT QA model
        t.add_computed_column(
            answer=question_answering(t.question, t.context, model_id='distilbert-base-cased-distilled-squad')
        )

        validate_update_status(t.insert(context=context, question=question), expected_rows=1)
        result = t.select(t.answer).collect()[0]

        # Verify we got an answer
        assert isinstance(result['answer'], dict)
        assert 'answer' in result['answer']
        assert 'score' in result['answer']
        assert 'paris' in result['answer']['answer'].lower()

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_translation(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import translation

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        english_text = 'Hello, how are you?'

        # Test with Helsinki-NLP translation model
        t.add_computed_column(
            french=translation(t.text, model_id='Helsinki-NLP/opus-mt-en-fr', src_lang='en', tgt_lang='fr')
        )

        validate_update_status(t.insert(text=english_text), expected_rows=1)
        result = t.select(t.french).collect()[0]

        # Verify we got a translation
        assert isinstance(result['french'], str)
        assert len(result['french'].strip()) > 0
        assert result['french'] != english_text  # Should be different from input

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_named_entity_recognition(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import named_entity_recognition

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        text_with_entities = 'Apple Inc. is located in Cupertino, California.'

        # Test with BERT NER model
        t.add_computed_column(
            entities=named_entity_recognition(
                t.text, model_id='dbmdz/bert-large-cased-finetuned-conll03-english', aggregation_strategy='simple'
            )
        )

        validate_update_status(t.insert(text=text_with_entities), expected_rows=1)
        result = t.select(t.entities).collect()[0]

        # Verify we got entities
        assert isinstance(result['entities'], list)
        assert len(result['entities']) > 0
        for entity in result['entities']:
            assert 'entity_group' in entity
            assert 'score' in entity
            assert 'word' in entity

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    @pytest.mark.skipif(sys.version_info >= (3, 13), reason='Not working on Python 3.13+')
    def test_automatic_speech_recognition(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import automatic_speech_recognition

        t = pxt.create_table('test_tbl', {'audio': pxt.Audio})
        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )

        # Test with Whisper model
        t.add_computed_column(transcript=automatic_speech_recognition(t.audio, model_id='openai/whisper-tiny'))

        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        result = t.select(t.transcript).collect()[0]

        # Verify we got a transcription
        assert isinstance(result['transcript'], str)
        assert len(result['transcript'].strip()) > 0

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_text_to_speech(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import text_to_speech

        t = pxt.create_table('test_tbl', {'text': pxt.String})
        test_text = 'Hello world, this is a test.'

        # Test with SpeechT5 model
        t.add_computed_column(
            audio=text_to_speech(t.text, model_id='microsoft/speecht5_tts', vocoder='microsoft/speecht5_hifigan')
        )

        validate_update_status(t.insert(text=test_text), expected_rows=1)
        result = t.select(t.audio).collect()[0]

        # Verify we got audio data
        assert result['audio'] is not None
        # Audio should be pxt.Audio type - basic check that it's not empty

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_text_to_image(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        from pixeltable.functions.huggingface import text_to_image

        t = pxt.create_table('test_tbl', {'prompt': pxt.String})
        test_prompt = 'a simple red circle'

        # Test with Stable Diffusion (use small image size for faster testing)
        t.add_computed_column(
            image=text_to_image(
                t.prompt,
                model_id='runwayml/stable-diffusion-v1-5',
                height=256,
                width=256,
                num_inference_steps=10,  # Fewer steps for testing
            )
        )

        validate_update_status(t.insert(prompt=test_prompt), expected_rows=1)
        result = t.select(t.image).collect()[0]

        # Verify we got an image
        assert result['image'] is not None
        # Should be a PIL Image or similar

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_image_to_image(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        from pixeltable.functions.huggingface import image_to_image

        t = pxt.create_table('test_tbl', {'img': pxt.Image, 'prompt': pxt.String})
        test_prompt = 'turn this into a red circle'

        # Test with Stable Diffusion
        t.add_computed_column(
            modified_image=image_to_image(
                t.img,
                t.prompt,
                model_id='runwayml/stable-diffusion-v1-5',
                strength=0.5,
                num_inference_steps=10,  # Fewer steps for testing
            )
        )

        validate_update_status(t.insert(img=SAMPLE_IMAGE_URL, prompt=test_prompt), expected_rows=1)
        result = t.select(t.modified_image).collect()[0]

        # Verify we got a modified image
        assert result['modified_image'] is not None
        # Should be a PIL Image or similar

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    @pytest.mark.skipif(sys.version_info >= (3, 13), reason='Not working on Python 3.13+')
    def test_image_to_video(self, reset_db: None) -> None:
        skip_test_if_not_installed('transformers')
        skip_test_if_not_installed('diffusers')
        skip_test_if_not_installed('av')
        from pixeltable.functions.huggingface import image_to_video

        t = pxt.create_table('test_tbl', {'img': pxt.Image})

        # Test with I2VGen-XL (use minimal settings for testing)
        t.add_computed_column(
            video=image_to_video(
                t.img,
                model_id='ali-vilab/i2vgen-xl',
                num_frames=8,  # Minimal frames for testing
                num_inference_steps=5,  # Fewer steps for testing
            )
        )

        validate_update_status(t.insert(img=SAMPLE_IMAGE_URL), expected_rows=1)
        result = t.select(t.video).collect()[0]

        # Verify we got a video
        assert result['video'] is not None
        # Should be pxt.Video type

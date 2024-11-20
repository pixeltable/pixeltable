import sysconfig
from typing import Any

import pytest

import pixeltable as pxt

from ..utils import (SAMPLE_IMAGE_URL, get_audio_files, get_image_files, get_sentences, reload_catalog, skip_test_if_not_installed,
                     validate_update_status)


class TestHuggingface:
    def test_hf_function(self, reset_db) -> None:
        skip_test_if_not_installed('sentence_transformers')
        t = pxt.create_table('test_tbl', {'input': pxt.String, 'bool_col': pxt.Bool})
        from pixeltable.functions.huggingface import sentence_transformer

        model_id = 'intfloat/e5-large-v2'
        t.add_column(e5=sentence_transformer(t.input, model_id=model_id))
        sents = get_sentences()
        status = t.insert({'input': s, 'bool_col': True} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # verify handling of constant params
        with pytest.raises(ValueError) as exc_info:
            t.add_column(e5_2=sentence_transformer(t.input, model_id=t.input))
        assert ': parameter model_id must be a constant value' in str(exc_info.value)
        with pytest.raises(ValueError) as exc_info:
            t.add_column(e5_2=sentence_transformer(t.input, model_id=model_id, normalize_embeddings=t.bool_col))
        assert ': parameter normalize_embeddings must be a constant value' in str(exc_info.value)

        # make sure this doesn't cause an exception
        # TODO: is there some way to capture the output?
        t.describe()

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_sentence_transformer(self, reset_db) -> None:
        skip_test_if_not_installed('sentence_transformers')
        t = pxt.create_table('test_tbl', {'input': pxt.String, 'input_list': pxt.Json})
        sents = get_sentences(10)
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        from pixeltable.functions.huggingface import sentence_transformer, sentence_transformer_list

        model_ids = ['sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-reranker-base']
        num_dims = [768, 768]
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed{idx}'
            t[col_name] = sentence_transformer(t.input, model_id=model_id, normalize_embeddings=True)
            assert t._schema[col_name].is_array_type()
            list_col_name = f'embed_list{idx}'
            t[list_col_name] = sentence_transformer_list(t.input_list, model_id=model_id, normalize_embeddings=True)
            assert t._schema[list_col_name] == pxt.JsonType(nullable=True)

        def verify_row(row: dict[str, Any]) -> None:
            for idx, (_, d) in enumerate(zip(model_ids, num_dims)):
                assert row[f'embed{idx}'].shape == (d,)
                assert len(row[f'embed_list{idx}']) == len(sents)
                assert all(len(v) == d for v in row[f'embed_list{idx}'])

        verify_row(t.tail(1)[0])

        # execution still works after reload
        reload_catalog()
        t = pxt.get_table('test_tbl')
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_cross_encoder(self, reset_db) -> None:
        skip_test_if_not_installed('sentence_transformers')
        t = pxt.create_table('test_tbl', {'input': pxt.String, 'input_list': pxt.Json})
        sents = get_sentences(10)
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        from pixeltable.functions.huggingface import cross_encoder, cross_encoder_list

        model_ids = ['cross-encoder/ms-marco-MiniLM-L-6-v2', 'cross-encoder/ms-marco-TinyBERT-L-2-v2']
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed{idx}'
            t[col_name] = cross_encoder(t.input, t.input, model_id=model_id)
            assert t._schema[col_name] == pxt.FloatType(nullable=True)
            list_col_name = f'embed_list{idx}'
            t[list_col_name] = cross_encoder_list(t.input, t.input_list, model_id=model_id)
            assert t._schema[list_col_name] == pxt.JsonType(nullable=True)

        def verify_row(row: dict[str, Any]) -> None:
            for i in range(len(model_ids)):
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

    def test_clip(self, reset_db) -> None:
        skip_test_if_not_installed('transformers')
        t = pxt.create_table('test_tbl', {'text': pxt.String, 'img': pxt.Image})
        num_rows = 10
        sents = get_sentences(num_rows)
        imgs = get_image_files()[:num_rows]
        status = t.insert({'text': text, 'img': img} for text, img in zip(sents, imgs))
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        from pixeltable.functions.huggingface import clip_image, clip_text

        model_ids = ['openai/clip-vit-base-patch32', 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K']
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed_text{idx}'
            t[col_name] = clip_text(t.text, model_id=model_id)
            assert t._schema[col_name].is_array_type()
            col_name = f'embed_img{idx}'
            t[col_name] = clip_image(t.img, model_id=model_id)
            assert t._schema[col_name].is_array_type()

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

    def test_detr_for_object_detection(self, reset_db) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import detr_for_object_detection
        from pixeltable.utils import coco

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t['detect'] = detr_for_object_detection(t.img, model_id='facebook/detr-resnet-50', threshold=0.8)
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

    def test_vit_for_image_classification(self, reset_db) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import vit_for_image_classification

        t = pxt.create_table('test_tbl', {'img': pxt.Image})
        t['img_class'] = vit_for_image_classification(t.img, model_id='google/vit-base-patch16-224', top_k=3)
        validate_update_status(t.insert(img=SAMPLE_IMAGE_URL), expected_rows=1)
        result = t.select(t.img_class).collect()[0]['img_class']
        assert result['labels'] == [962, 935, 937]
        assert result['label_text'] == ['meat loaf, meatloaf', 'mashed potato', 'broccoli']

    @pytest.mark.skipif(sysconfig.get_platform() == 'linux-aarch64', reason='Not supported on Linux ARM')
    def test_speech2text_for_conditional_generation(self, reset_db) -> None:
        skip_test_if_not_installed('transformers')
        from pixeltable.functions.huggingface import speech2text_for_conditional_generation

        t = pxt.create_table('test_tbl', {'audio': pxt.Audio})
        audio_file = next(
            file for file in get_audio_files() if file.endswith('jfk_1961_0109_cityuponahill-excerpt.flac')
        )
        t['transcription'] = speech2text_for_conditional_generation(t.audio, model_id='facebook/s2t-small-librispeech-asr')
        t['translation'] = speech2text_for_conditional_generation(
            t.audio, model_id='facebook/s2t-medium-mustc-multilingual-st', language='fr')

        validate_update_status(t.insert(audio=audio_file), expected_rows=1)
        result = t.collect()
        assert 'administration' in result['transcription'][0]
        assert 'construire' in result['translation'][0]

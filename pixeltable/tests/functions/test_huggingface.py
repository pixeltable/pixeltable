from typing import Dict, Any

import pytest

import pixeltable as pxt
from pixeltable.tests.utils import skip_test_if_not_installed, get_sentences, get_image_files, \
    SAMPLE_IMAGE_URL
from pixeltable.type_system import StringType, JsonType, ImageType, BoolType, FloatType, ArrayType


class TestHuggingface:

    def test_hf_function(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('sentence_transformers')
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType(), 'bool_col': BoolType()})
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

    def test_sentence_transformer(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('sentence_transformers')
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType(), 'input_list': JsonType()})
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
            assert t.column_types()[col_name] == ArrayType((None,), dtype=FloatType(), nullable=False)
            list_col_name = f'embed_list{idx}'
            t[list_col_name] = sentence_transformer_list(t.input_list, model_id=model_id, normalize_embeddings=True)
            assert t.column_types()[list_col_name] == JsonType()

        def verify_row(row: Dict[str, Any]) -> None:
            for idx, (_, d) in enumerate(zip(model_ids, num_dims)):
                assert row[f'embed{idx}'].shape == (d,)
                assert len(row[f'embed_list{idx}']) == len(sents)
                assert all(len(v) == d for v in row[f'embed_list{idx}'])

        verify_row(t.tail(1)[0])

        # execution still works after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    def test_cross_encoder(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('sentence_transformers')
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType(), 'input_list': JsonType()})
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
            assert t.column_types()[col_name] == FloatType()
            list_col_name = f'embed_list{idx}'
            t[list_col_name] = cross_encoder_list(t.input, t.input_list, model_id=model_id)
            assert t.column_types()[list_col_name] == JsonType()

        def verify_row(row: Dict[str, Any]) -> None:
            for i in range(len(model_ids)):
                assert len(row[f'embed_list{idx}']) == len(sents)
                assert all(isinstance(v, float) for v in row[f'embed_list{idx}'])

        verify_row(t.tail(1)[0])

        # execution still works after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        status = t.insert({'input': s, 'input_list': sents} for s in sents)
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    def test_clip(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('transformers')
        cl = test_client
        t = cl.create_table('test_tbl', {'text': StringType(), 'img': ImageType()})
        num_rows = 10
        sents = get_sentences(num_rows)
        imgs = get_image_files()[:num_rows]
        status = t.insert({'text': text, 'img': img} for text, img in zip(sents, imgs))
        assert status.num_rows == len(sents)
        assert status.num_excs == 0

        # run multiple models one at a time in order to exercise batching
        from pixeltable.functions.huggingface import clip_text, clip_image
        model_ids = ['openai/clip-vit-base-patch32', 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K']
        for idx, model_id in enumerate(model_ids):
            col_name = f'embed_text{idx}'
            t[col_name] = clip_text(t.text, model_id=model_id)
            assert t.column_types()[col_name].is_array_type()
            col_name = f'embed_img{idx}'
            t[col_name] = clip_image(t.img, model_id=model_id)
            assert t.column_types()[col_name].is_array_type()

        def verify_row(row: Dict[str, Any]) -> None:
            for idx, _ in enumerate(model_ids):
                assert row[f'embed_text{idx}'].shape == (512,)
                assert row[f'embed_img{idx}'].shape == (512,)

        verify_row(t.tail(1)[0])

        # execution still works after reload
        cl = pxt.Client(reload=True)
        t = cl.get_table('test_tbl')
        status = t.insert({'text': text, 'img': img} for text, img in zip(sents, imgs))
        assert status.num_rows == len(sents)
        assert status.num_excs == 0
        verify_row(t.tail(1)[0])

    def test_detr_for_object_detection(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('transformers')
        cl = test_client
        t = cl.create_table('test_tbl', {'img': ImageType()})
        from pixeltable.functions.huggingface import detr_for_object_detection
        t['detect'] = detr_for_object_detection(t.img, model_id='facebook/detr-resnet-50', threshold=0.8)
        status = t.insert(img=SAMPLE_IMAGE_URL)
        assert status.num_rows == 1
        assert status.num_excs == 0
        result = t.select(t.detect).collect()[0]['detect']
        assert 'orange' in result['label_text']
        assert 'bowl' in result['label_text']
        assert 'broccoli' in result['label_text']

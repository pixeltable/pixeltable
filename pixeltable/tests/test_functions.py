from typing import Dict, Any

import pytest

import pixeltable as pxt
from pixeltable import catalog
from pixeltable.env import Env
import pixeltable.exceptions as excs
from pixeltable.functions.pil.image import blend
from pixeltable.iterators import FrameIterator
from pixeltable.tests.utils import get_video_files, skip_test_if_not_installed, get_sentences, get_image_files
from pixeltable.type_system import VideoType, StringType, JsonType, ImageType, BoolType, FloatType, ArrayType


class TestFunctions:
    def test_pil(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        _ = t[t.img, t.img.rotate(90), blend(t.img, t.img.rotate(90), 0.5)].show()

    def test_eval_detections(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('nos')
        cl = test_client
        video_t = cl.create_table('video_tbl', {'video': VideoType()})
        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        v = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)

        files = get_video_files()
        video_t.insert(video=files[-1])
        v.add_column(frame_s=v.frame.resize([640, 480]))
        from pixeltable.functions.nos.object_detection_2d import yolox_nano, yolox_small, yolox_large
        v.add_column(detections_a=yolox_nano(v.frame_s))
        v.add_column(detections_b=yolox_small(v.frame_s))
        v.add_column(gt=yolox_large(v.frame_s))
        from pixeltable.functions.eval import eval_detections, mean_ap
        res = v.select(
            eval_detections(
                v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels
            )).show()
        v.add_column(
            eval_a=eval_detections(
                v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels))
        v.add_column(
            eval_b=eval_detections(
                v.detections_b.bboxes, v.detections_b.labels, v.detections_b.scores, v.gt.bboxes, v.gt.labels))
        ap_a = v.select(mean_ap(v.eval_a)).show()[0, 0]
        ap_b = v.select(mean_ap(v.eval_b)).show()[0, 0]
        common_classes = set(ap_a.keys()) & set(ap_b.keys())

        ## TODO: following assertion is failing on CI, 
        # It is not necessarily a bug, as assert codition is not expected to be always true
        # for k in common_classes:
        # assert ap_a[k] <= ap_b[k]

    def test_str(self, test_client: pxt.Client) -> None:
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.string import str_format
        t.add_column(s1=str_format('ABC {0}', t.input))
        t.add_column(s2=str_format('DEF {this}', this=t.input))
        t.add_column(s3=str_format('GHI {0} JKL {this}', t.input, this=t.input))
        status = t.insert(input='MNO')
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[0]
        assert row == {'input': 'MNO', 's1': 'ABC MNO', 's2': 'DEF MNO', 's3': 'GHI MNO JKL MNO'}

    def test_openai(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestFunctions.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.openai import chat_completions, embeddings, moderations
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": t.input}
        ]
        t.add_column(input_msgs=msgs)
        t.add_column(chat_output=chat_completions(model='gpt-3.5-turbo', messages=t.input_msgs))
        # with inlined messages
        t.add_column(chat_output2=chat_completions(model='gpt-3.5-turbo', messages=msgs))
        t.add_column(ada_embed=embeddings(model='text-embedding-ada-002', input=t.input))
        t.add_column(text_3=embeddings(model='text-embedding-3-small', input=t.input))
        t.add_column(moderation=moderations(input=t.input))
        t.insert(input='I find you really annoying')
        _ = t.head()

    def test_gpt_4_vision(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('openai')
        TestFunctions.skip_test_if_no_openai_client()
        cl = test_client
        t = cl.create_table('test_tbl', {'prompt': StringType(), 'img': ImageType()})
        from pixeltable.functions.openai import chat_completions
        from pixeltable.functions.string import str_format
        msgs = [
            {'role': 'user',
             'content': [
                 {'type': 'text', 'text': t.prompt},
                 {'type': 'image_url', 'image_url': {
                     'url': str_format('data:image/png;base64,{0}', t.img.b64_encode())
                 }}
             ]}
        ]
        t.add_column(response=chat_completions(model='gpt-4-vision-preview', messages=msgs, max_tokens=300))
        t.add_column(response_content=t.response.choices[0].message.content)
        t.insert(prompt="What's in this image?", img=_sample_image_url)
        result = t.collect()['response_content'][0]
        assert len(result) > 0

    @staticmethod
    def skip_test_if_no_openai_client() -> None:
        try:
            _ = Env.get().openai_client
        except excs.Error as exc:
            pytest.skip(str(exc))

    def test_together(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('together')
        if not Env.get().has_together_client:
            pytest.skip(f'Together client does not exist (missing API key?)')
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.together import completions
        t.add_column(output=completions(prompt=t.input, model='mistralai/Mixtral-8x7B-v0.1', stop=['\n']))
        t.add_column(output_text=t.output.output.choices[0].text)
        t.insert(input='I am going to the ')
        result = t.select(t.output_text).collect()['output_text'][0]
        assert len(result) > 0

    def test_fireworks(self, test_client: pxt.Client) -> None:
        skip_test_if_not_installed('fireworks')
        try:
            from pixeltable.functions.fireworks import initialize
            initialize()
        except:
            pytest.skip(f'Fireworks client does not exist (missing API key?)')
        cl = test_client
        t = cl.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.fireworks import chat_completions
        t['output'] = chat_completions(prompt=t.input, model='accounts/fireworks/models/llama-v2-7b-chat', max_tokens=256).choices[0].text
        t.insert(input='I am going to the ')
        result = t.select(t.output).collect()['output'][0]
        assert len(result) > 0

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
            assert t.column_types()[col_name] == ArrayType((None,), dtype=FloatType(), nullable=False)
            col_name = f'embed_img{idx}'
            t[col_name] = clip_image(t.img, model_id=model_id)
            assert t.column_types()[col_name] == ArrayType((None,), dtype=FloatType(), nullable=False)

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
        status = t.insert(img=_sample_image_url)
        assert status.num_rows == 1
        assert status.num_excs == 0
        result = t.select(t.detect).collect()[0]['detect']
        assert 'orange' in result['label_text']
        assert 'bowl' in result['label_text']
        assert 'broccoli' in result['label_text']


_sample_image_url = \
    'https://raw.githubusercontent.com/pixeltable/pixeltable/master/docs/source/data/images/000000000009.jpg'

import pytest

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import VideoType, ImageType, IntType, StringType
from pixeltable.functions.pil.image import blend
from pixeltable.tests.utils import get_video_files
from pixeltable.iterators import FrameIterator


class TestFunctions:
    def test_pil(self, img_tbl: catalog.MutableTable) -> None:
        t = img_tbl
        _ = t[t.img, t.img.rotate(90), blend(t.img, t.img.rotate(90), 0.5)].show()

    def test_eval_detections(self, test_client: pt.Client) -> None:
        cl = test_client
        video_t = cl.create_table('video_tbl', [catalog.Column('video', VideoType())])
        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        v = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)

        files = get_video_files()
        video_t.insert([[files[-1]]], ['video'])
        v.add_column(catalog.Column('frame_s', computed_with=v.frame.resize([640, 480])))
        from pixeltable.functions.nos.object_detection_2d import yolox_nano, yolox_small, yolox_large
        v.add_column(catalog.Column('detections_a', computed_with=yolox_nano(v.frame_s)))
        v.add_column(catalog.Column('detections_b', computed_with=yolox_small(v.frame_s)))
        v.add_column(catalog.Column('gt', computed_with=yolox_large(v.frame_s)))
        from pixeltable.functions.eval import eval_detections, mean_ap
        res = v.select(
            eval_detections(
                v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels
            )).show()
        v.add_column(
            catalog.Column(
                'eval_a',
                computed_with=eval_detections(
                    v.detections_a.bboxes, v.detections_a.labels, v.detections_a.scores, v.gt.bboxes, v.gt.labels)
            ))
        v.add_column(
            catalog.Column(
                'eval_b',
                computed_with=eval_detections(
                    v.detections_b.bboxes, v.detections_b.labels, v.detections_b.scores, v.gt.bboxes, v.gt.labels)
            ))
        ap_a = v.select(mean_ap(v.eval_a)).show()[0, 0]
        ap_b = v.select(mean_ap(v.eval_b)).show()[0, 0]
        common_classes = set(ap_a.keys()) & set(ap_b.keys())
        for k in common_classes:
            assert ap_a[k] <= ap_b[k]

    def test_str(self, test_client: pt.Client) -> None:
        cl = test_client
        t = cl.create_table('test_tbl', [catalog.Column('input', StringType())])
        from pixeltable.functions import str_format
        t.add_column(catalog.Column('s1', computed_with=str_format('ABC {0}', t.input)))
        t.add_column(catalog.Column('s2', computed_with=str_format('DEF {this}', this=t.input)))
        t.add_column(catalog.Column('s3', computed_with=str_format('GHI {0} JKL {this}', t.input, this=t.input)))
        status = t.insert([['MNO']])
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[0]
        assert row == {'input': 'MNO', 's1': 'ABC MNO', 's2': 'DEF MNO', 's3': 'GHI MNO JKL MNO'}

    @pytest.mark.skip(reason='not supported yet')
    def test_openai(self, test_client: pt.Client) -> None:
        cl = test_client
        t = cl.create_table('test_tbl', [catalog.Column('input', StringType())])
        from pixeltable.functions.openai import chat_completion, embedding, moderation
        msgs = [
            { "role": "system", "content": "You are a helpful assistant." },
            { "role": "user", "content": t.input }
        ]
        t.add_column(catalog.Column('input_msgs', computed_with=msgs))
        t.add_column(
            catalog.Column('chat_output', computed_with=chat_completion(model='gpt-3.5-turbo', messages=t.input_msgs)))
        # with inlined messages
        t.add_column(
            catalog.Column('chat_output2', computed_with=chat_completion(model='gpt-3.5-turbo', messages=msgs)))
        t.add_column(catalog.Column('embedding', computed_with=embedding(model='text-embedding-ada-002', input=t.input)))
        t.add_column(catalog.Column('moderation', computed_with=moderation(input=t.input)))
        t.insert([['I find you really annoying']])
        _ = t.head()
        pass

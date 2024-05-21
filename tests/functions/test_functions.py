import pixeltable as pxt
from pixeltable.iterators import FrameIterator
from pixeltable.type_system import VideoType, StringType

from ..utils import get_video_files, skip_test_if_not_installed


class TestFunctions:

    def test_eval_detections(self, reset_db) -> None:
        skip_test_if_not_installed('yolox')
        from pixeltable.ext.functions.yolox import yolox
        video_t = pxt.create_table('video_tbl', {'video': VideoType()})
        # create frame view
        v = pxt.create_view('test_view', video_t, iterator=FrameIterator.create(video=video_t.video, fps=1))

        files = get_video_files()
        video_t.insert(video=files[-1])
        v.add_column(frame_s=v.frame.resize([640, 480]))
        v.add_column(detections_a=yolox(v.frame_s, model_id='yolox_nano'))
        v.add_column(detections_b=yolox(v.frame_s, model_id='yolox_s'))
        v.add_column(gt=yolox(v.frame_s, model_id='yolox_l'))
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
        _ = v.select(mean_ap(v.eval_a)).show()[0, 0]
        _ = v.select(mean_ap(v.eval_b)).show()[0, 0]

    def test_str(self, reset_db) -> None:
        t = pxt.create_table('test_tbl', {'input': StringType()})
        from pixeltable.functions.string import str_format
        t.add_column(s1=str_format('ABC {0}', t.input))
        t.add_column(s2=str_format('DEF {this}', this=t.input))
        t.add_column(s3=str_format('GHI {0} JKL {this}', t.input, this=t.input))
        status = t.insert(input='MNO')
        assert status.num_rows == 1
        assert status.num_excs == 0
        row = t.head()[0]
        assert row == {'input': 'MNO', 's1': 'ABC MNO', 's2': 'DEF MNO', 's3': 'GHI MNO JKL MNO'}

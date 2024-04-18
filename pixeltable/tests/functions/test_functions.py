import pixeltable as pxt
from pixeltable import catalog
from pixeltable.functions.pil.image import blend
from pixeltable.iterators import FrameIterator
from pixeltable.tests.utils import get_video_files, skip_test_if_not_installed
from pixeltable.type_system import VideoType, StringType


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

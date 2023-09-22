import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import VideoType, ImageType, IntType
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
        v.add_column(catalog.Column('frame_s', computed_with=v.frame.resize((640, 480))))
        from pixeltable.functions.object_detection_2d import yolox_nano, yolox_small, yolox_large
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

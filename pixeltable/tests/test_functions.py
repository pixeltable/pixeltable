import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import VideoType, ImageType, IntType
from pixeltable.functions.pil.image import blend
from pixeltable.tests.utils import get_video_files


class TestFunctions:
    def test_pil(self, img_tbl: catalog.Table) -> None:
        t = img_tbl
        _ = t[t.img, t.img.rotate(90), blend(t.img, t.img.rotate(90), 0.5)].show()

    def test_eval_detections(self, test_client: pt.Client) -> None:
        cl = test_client
        cols = [
            catalog.Column('video', VideoType()),
            catalog.Column('frame', ImageType()),
            catalog.Column('frame_idx', IntType()),
        ]
        tbl = cl.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=1)
        files = get_video_files()
        tbl.insert([[files[-1]]], ['video'])
        tbl.add_column(catalog.Column('frame_s', computed_with=tbl.frame.resize((640, 480))))
        from pixeltable.functions.object_detection_2d import yolox_nano, yolox_large, yolox_xlarge
        tbl.add_column(catalog.Column('detections_n', computed_with=yolox_nano(tbl.frame_s)))
        tbl.add_column(catalog.Column('detections_l', computed_with=yolox_large(tbl.frame_s)))
        tbl.add_column(catalog.Column('gt', computed_with=yolox_xlarge(tbl.frame_s)))
        from pixeltable.functions.eval import eval_detections, mean_ap
        res = tbl.select(
            eval_detections(
                tbl.detections_n.bboxes, tbl.detections_n.labels, tbl.detections_n.scores, tbl.gt.bboxes, tbl.gt.labels
            )).show()
        tbl.add_column(
            catalog.Column(
                'eval_n',
                computed_with=eval_detections(
                    tbl.detections_n.bboxes, tbl.detections_n.labels, tbl.detections_n.scores, tbl.gt.bboxes,
                    tbl.gt.labels)
            ))
        tbl.add_column(
            catalog.Column(
                'eval_l',
                computed_with=eval_detections(
                    tbl.detections_l.bboxes, tbl.detections_l.labels, tbl.detections_l.scores, tbl.gt.bboxes,
                    tbl.gt.labels)
            ))
        ap_n = tbl.select(mean_ap(tbl.eval_n)).show()[0, 0]
        ap_l = tbl.select(mean_ap(tbl.eval_l)).show()[0, 0]
        common_classes = set(ap_n.keys()) & set(ap_l.keys())
        for k in common_classes:
            assert ap_n[k] <= ap_l[k]

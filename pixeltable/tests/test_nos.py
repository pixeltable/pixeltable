import pytest

from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import get_video_files
from pixeltable.exprs import Literal
import pixeltable as pt


class TestNOS:
    def test_basic(self, test_client: pt.Client) -> None:
        cl = test_client
        cols = [
            catalog.Column('video', VideoType()),
            catalog.Column('frame', ImageType()),
            catalog.Column('frame_idx', IntType()),
        ]
        tbl = cl.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=1)
        tbl.add_column(catalog.Column('transform1', computed_with=tbl.frame.rotate(30), stored=False))
        from pixeltable.functions.object_detection_2d import \
            torchvision_fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn
        tbl.add_column(catalog.Column('detections', computed_with=fasterrcnn(tbl.transform1)))
        from pixeltable.functions.image_embedding import openai_clip
        tbl.add_column(catalog.Column('embed', computed_with=openai_clip(tbl.transform1.resize((224, 224)))))
        # add a stored column that isn't referenced in nos calls
        tbl.add_column(catalog.Column('transform2', computed_with=tbl.frame.rotate(60), stored=True))

        tbl.insert([[get_video_files()[0]]], ['video'])

    def test_exceptions(self, test_client: pt.Client) -> None:
        cl = test_client
        cols = [
            catalog.Column('video', VideoType()),
            catalog.Column('frame', ImageType()),
            catalog.Column('frame_idx', IntType()),
        ]
        tbl = cl.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=1)
        tbl.insert([[get_video_files()[0]]], ['video'])
        tbl.add_column(catalog.Column('frame_s', computed_with=tbl.frame.resize((640, 480))))
        # 'rotated' has exceptions
        tbl.add_column(catalog.Column(
            'rotated', ImageType(), computed_with=lambda frame_s, frame_idx: frame_s.rotate(int(360 / frame_idx))))
        from pixeltable.functions.object_detection_2d import yolox_medium
        tbl.add_column(catalog.Column('detections', computed_with=yolox_medium(tbl.rotated), stored=True))
        assert tbl[tbl.detections.errortype != None].count() == 1

    @pytest.mark.skip(reason='too slow')
    def test_sd(self, test_client: pt.Client) -> None:
        """Test model that mixes batched with scalar parameters"""
        t = test_client.create_table('sd_test', [pt.Column('prompt', pt.StringType())])
        t.insert([['cat on a sofa']])
        from pixeltable.functions.image_generation import stabilityai_stable_diffusion_2 as sd2
        t.add_column(pt.Column('img', computed_with=sd2(t.prompt, 1, 512, 512), stored=True))
        img = t[t.img].show(1)[0, 0]
        assert img.size == (512, 512)

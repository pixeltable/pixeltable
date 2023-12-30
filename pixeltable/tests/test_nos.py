import pytest

import pixeltable as pt
from pixeltable import catalog
from pixeltable.type_system import ImageType, VideoType
from pixeltable.tests.utils import get_video_files
from pixeltable.iterators import FrameIterator


class TestNOS:
    def test_basic(self, test_client: pt.Client) -> None:
        cl = test_client
        video_t = cl.create_table('video_tbl', [catalog.Column('video', VideoType())])
        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        v = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        v.add_column(catalog.Column('transform1', computed_with=v.frame.rotate(30), stored=False))
        from pixeltable.functions.nos.object_detection_2d import \
            torchvision_fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn
        v.add_column(catalog.Column('detections', computed_with=fasterrcnn(v.transform1)))
        from pixeltable.functions.nos.image_embedding import openai_clip
        v.add_column(catalog.Column('embed', computed_with=openai_clip(v.transform1.resize([224, 224]))))
        # add a stored column that isn't referenced in nos calls
        v.add_column(catalog.Column('transform2', computed_with=v.frame.rotate(60), stored=True))

        status = video_t.insert([{'video': get_video_files()[0]}])
        pass

    def test_exceptions(self, test_client: pt.Client) -> None:
        cl = test_client
        video_t = cl.create_table('video_tbl', [catalog.Column('video', VideoType())])
        # create frame view
        args = {'video': video_t.video, 'fps': 1}
        v = cl.create_view('test_view', video_t, iterator_class=FrameIterator, iterator_args=args)
        video_t.insert([{'video': get_video_files()[0]}])

        v.add_column(catalog.Column('frame_s', computed_with=v.frame.resize([640, 480])))
        # 'rotated' has exceptions
        v.add_column(catalog.Column(
            'rotated', ImageType(), computed_with=lambda frame_s, frame_idx: frame_s.rotate(int(360 / frame_idx))))
        from pixeltable.functions.nos.object_detection_2d import yolox_medium
        v.add_column(catalog.Column('detections', computed_with=yolox_medium(v.rotated), stored=True))
        assert v.where(v.detections.errortype != None).count() == 1

    @pytest.mark.skip(reason='too slow')
    def test_sd(self, test_client: pt.Client) -> None:
        """Test model that mixes batched with scalar parameters"""
        t = test_client.create_table('sd_test', [pt.Column('prompt', pt.StringType())])
        t.insert([{'prompt': 'cat on a sofa'}])
        from pixeltable.functions.nos.image_generation import stabilityai_stable_diffusion_2 as sd2
        t.add_column(pt.Column('img', computed_with=sd2(t.prompt, 1, 512, 512), stored=True))
        img = t[t.img].show(1)[0, 0]
        assert img.size == (512, 512)

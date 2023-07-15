from pixeltable import catalog
from pixeltable.type_system import \
    StringType, IntType, FloatType, TimestampType, ImageType, VideoType, JsonType, BoolType, ArrayType
from pixeltable.tests.utils import get_video_files
from pixeltable.function import FunctionRegistry
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
        tbl.add_column(catalog.Column('embed', computed_with=openai_clip(tbl.transform1)))
        # add a stored column that isn't referenced in nos calls
        tbl.add_column(catalog.Column('transform2', computed_with=tbl.frame.rotate(60), stored=True))

        tbl.insert_rows([[get_video_files()[0]]], ['video'])

    # def test_demo(self, test_db: catalog.Db) -> None:
    #     db = test_db
    #     cols = [
    #         catalog.Column('video', VideoType(), nullable=False),
    #         catalog.Column('frame', ImageType(), nullable=False),
    #         catalog.Column('frame_idx', IntType(), nullable=False),
    #     ]
    #     tbl = db.create_table(
    #         'test', cols, extract_frames_from='video', extracted_frame_col='frame',
    #         extracted_frame_idx_col='frame_idx', extracted_fps=1)
    #     from pixeltable.functions.object_detection_2d import \
    #         torchvision_fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn
    #     tbl.add_column(catalog.Column('detections', computed_with=fasterrcnn(tbl.frame)))
    #
    #     tbl.insert_rows([[get_video_files()[1]]], ['video'])

import pytest
import pandas as pd

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import utils


class TestVideo:
    def test_basic(self, test_db: pt.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video_file', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        df = pd.DataFrame({'video_file': video_filepaths})
        tbl = db.create_table('test', cols)
        tbl.insert_pandas(df, video_column='video_file', frame_column='frame', frame_idx_column='frame_idx', fps=1)
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()
        _ = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(0)

        tbl.insert_pandas(df, video_column='video_file', frame_column='frame', frame_idx_column='frame_idx', fps=1)
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()
        # revert() also removes extracted frames
        tbl.revert()
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()

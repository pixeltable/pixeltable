import pytest
import pandas as pd

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import utils
from pixeltable import exceptions as exc


class TestVideo:
    def test_basic(self, test_db: pt.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        tbl = db.create_table(
            'test', cols, extract_frames_from = 'video', extracted_frame_col = 'frame',
            extracted_frame_idx_col = 'frame_idx', extracted_fps = 1)
        tbl.insert_rows([[p] for p in video_filepaths], columns=['video'])
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()
        _ = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(0)

        # missing 'columns' arg
        with pytest.raises(exc.Error):
            tbl.insert_rows([[p] for p in video_filepaths])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2], [3]], columns=['video'])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2]], columns=['video'])

        # revert() also removes extracted frames
        tbl.revert()
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()

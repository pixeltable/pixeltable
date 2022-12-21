import pytest
import pandas as pd

import pixeltable as pt
import pixeltable.utils.video as v
from pixeltable.type_system import StringType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import env, catalog


class TestVideo:
    def test_basic(self, test_db: pt.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video_file', StringType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        df = pd.DataFrame({'video_file': video_filepaths})
        tbl = db.create_table('test', cols)
        tbl.insert_pandas(df, video_column='video_file', frame_column='frame', frame_idx_column='frame_idx', fps=1)
        df = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)]
        res = df.show(0)
        print(res)


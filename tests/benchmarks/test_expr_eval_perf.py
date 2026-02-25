from typing import Any

import pyarrow as pa
import pytest
import sqlalchemy as sa

import pixeltable as pxt
import pixeltable.functions.audio as pxt_audio
import pixeltable.functions.video as pxt_video
from pixeltable.env import Env
from pixeltable.functions.string import isalpha, isascii, lower, upper

from ..utils import SAMPLE_IMAGE_URL, get_audio_files, get_video_files


class TestExprEvalPerformance:
    """Benchmarks for expression evaluation dispatch optimization.

    These benchmarks include both SQL-pushdown functions (string operations)
    and UDFs that must go through ExprEvalNode (image, video, audio functions).
    """

    @pytest.mark.benchmark(group='batch_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_select_batch_scaling(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Test how performance scales with row count (vectorization benefit)."""
        t = pxt.create_table(f'scale_tbl_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        def select_with_functions() -> None:
            res = t.select(t.c1, isascii(t.c2), isalpha(t.c2)).collect()
            assert len(res) == row_count

        benchmark(select_with_functions)

    @pytest.mark.benchmark(group='batch_insert_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_insert_batch_scaling_pxt(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Benchmark pixeltable batch inserts with no computed columns."""
        t = pxt.create_table(f'insert_pxt_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        rows = [{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)]

        benchmark(t.insert, rows)

    @pytest.mark.benchmark(group='batch_insert_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_insert_batch_scaling_sql(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Benchmark raw SQLAlchemy batch inserts as a baseline comparison to pixeltable."""
        engine = Env.get().engine
        meta = sa.MetaData()
        raw_tbl = sa.Table(f'raw_insert_{row_count}', meta, sa.Column('c1', sa.Integer), sa.Column('c2', sa.String))
        meta.create_all(engine)
        rows = [{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)]

        def sql_insert() -> None:
            with engine.begin() as conn:
                conn.execute(raw_tbl.insert(), rows)

        try:
            benchmark(sql_insert)
        finally:
            meta.drop_all(engine)

    @pytest.mark.benchmark(group='batch_insert_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_insert_batch_scaling_pyarrow(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Benchmark PyArrow -> pandas -> SQL batch inserts as a second baseline."""
        engine = Env.get().engine
        table_name = f'pa_insert_{row_count}'
        arrow_tbl = pa.table(
            {
                'c1': pa.array(range(row_count), type=pa.int32()),
                'c2': pa.array([f'str_{i}' for i in range(row_count)], type=pa.string()),
            }
        )

        # Create the table once so repeated benchmark rounds just append
        arrow_tbl.to_pandas().head(0).to_sql(table_name, engine, if_exists='replace', index=False)

        def pyarrow_insert() -> None:
            arrow_tbl.to_pandas().to_sql(table_name, engine, if_exists='append', index=False)

        try:
            benchmark(pyarrow_insert)
        finally:
            with engine.begin() as conn:
                conn.execute(sa.text(f'DROP TABLE IF EXISTS {table_name}'))

    @pytest.mark.benchmark(group='image_transform')
    def test_insert_image_resize(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark image resize operations."""
        row_count = 200

        t = pxt.create_table('img_resize_tbl', {'img': pxt.Image})
        t.add_computed_column(resized=t.img.resize((128, 128)))

        def insert_resized() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_resized)

    @pytest.mark.benchmark(group='video_frames')
    def test_insert_video_frame_extraction(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark video frame extraction."""
        video_files = get_video_files()
        if not video_files:
            pytest.skip('No test videos available')
        video_path = video_files[0]

        t = pxt.create_table('video_frame_tbl', {'video': pxt.Video, 'timestamp': pxt.Float})
        t.add_computed_column(frame=pxt_video.extract_frame(t.video, timestamp=t.timestamp))

        def insert_frames() -> None:
            t.insert([{'video': video_path, 'timestamp': i * 0.5} for i in range(20)])

        benchmark(insert_frames)

    @pytest.mark.benchmark(group='audio_metadata')
    def test_insert_audio_metadata(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark audio metadata extraction."""
        audio_files = get_audio_files()
        if not audio_files:
            pytest.skip('No test audio available')
        audio_path = audio_files[0]

        t = pxt.create_table('audio_meta_tbl', {'audio': pxt.Audio})
        t.add_computed_column(metadata=pxt_audio.get_metadata(t.audio))

        def insert_metadata() -> None:
            t.insert([{'audio': audio_path} for _ in range(50)])

        benchmark(insert_metadata)

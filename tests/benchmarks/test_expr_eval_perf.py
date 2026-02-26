import pyarrow as pa
import pytest
import sqlalchemy as sa
from pytest_benchmark.fixture import BenchmarkFixture

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.env import Env

from ..utils import SAMPLE_IMAGE_URL, get_audio_files, get_video_files


@pxt.udf
def noop_str(s: str) -> bool:
    """Always returns True. Used in benchmarks to ensure expression evaluation goes through
    ExprEvalNode and cannot be pushed down to SQL."""
    return len(s) >= 0


class TestExprEvalPerformance:
    """Benchmarks for expression evaluation dispatch optimization.

    These benchmarks include both SQL-pushdown functions (string operations)
    and UDFs that must go through ExprEvalNode (image, video, audio functions).
    """

    @pytest.mark.benchmark(group='batch_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_select_batch_scaling(self, uses_db: None, benchmark: BenchmarkFixture, row_count: int) -> None:
        """Test how performance scales with row count (vectorization benefit)."""
        t = pxt.create_table(f'scale_tbl_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        rows = [{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)]
        t.insert(rows)

        def select_with_functions() -> None:
            res = t.select(t.c1, noop_str(t.c2)).collect()
            assert len(res) == row_count

        benchmark(select_with_functions)

    @pytest.mark.benchmark(group='batch_insert_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_insert_batch_scaling_pxt(self, uses_db: None, benchmark: BenchmarkFixture, row_count: int) -> None:
        """Benchmark pixeltable batch inserts with no computed columns."""
        t = pxt.create_table(f'insert_pxt_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        rows = [{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)]

        benchmark(t.insert, rows)

    @pytest.mark.benchmark(group='batch_insert_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_insert_batch_scaling_sql(self, uses_db: None, benchmark: BenchmarkFixture, row_count: int) -> None:
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
    def test_insert_batch_scaling_pyarrow(self, uses_db: None, benchmark: BenchmarkFixture, row_count: int) -> None:
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
    def test_insert_image_resize(self, uses_db: None, benchmark: BenchmarkFixture) -> None:
        """Benchmark image resize operations."""
        row_count = 200

        t = pxt.create_table('img_resize_tbl', {'img': pxt.Image})
        t.add_computed_column(resized=t.img.resize((128, 128)))
        rows = [{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)]

        def insert_resized() -> None:
            t.insert(rows)

        benchmark(insert_resized)

    @pytest.mark.benchmark(group='video_frames')
    def test_insert_video_frame_extraction(self, uses_db: None, benchmark: BenchmarkFixture) -> None:
        """Benchmark video frame extraction."""
        video_files = get_video_files()
        assert len(video_files) > 0
        video_path = video_files[0]

        t = pxt.create_table('video_frame_tbl', {'video': pxt.Video})
        pxt.create_view('video_frames', t, iterator=pxtf.video.frame_iterator(t.video))
        rows = [{'video': video_path}]

        def insert_frames() -> None:
            t.insert(rows)

        benchmark(insert_frames)

    @pytest.mark.benchmark(group='audio_metadata')
    def test_insert_audio_metadata(self, uses_db: None, benchmark: BenchmarkFixture) -> None:
        """Benchmark audio metadata extraction."""
        audio_files = get_audio_files()
        assert len(audio_files) > 0
        audio_path = audio_files[0]

        t = pxt.create_table('audio_meta_tbl', {'audio': pxt.Audio})
        t.add_computed_column(metadata=pxtf.audio.get_metadata(t.audio))
        rows = [{'audio': audio_path} for _ in range(50)]

        def insert_metadata() -> None:
            t.insert(rows)

        benchmark(insert_metadata)

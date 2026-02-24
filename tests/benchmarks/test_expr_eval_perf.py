from typing import Any

import pytest

import pixeltable as pxt
import pixeltable.functions.audio as pxt_audio
import pixeltable.functions.video as pxt_video
from pixeltable.functions.string import isalpha, isascii, lower, upper

from ..utils import SAMPLE_IMAGE_URL, get_audio_files, get_video_files


class TestExprEvalPerformance:
    """Benchmarks for expression evaluation dispatch optimization.

    These benchmarks include both SQL-pushdown functions (string operations)
    and UDFs that must go through ExprEvalNode (image, video, audio functions).
    """

    # -------------------------------------------------------------------------
    # Original String/SQL Benchmarks (may use SQL pushdown)
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='wide_table')
    def test_wide_table_evaluation(self, uses_db: None, benchmark: Any) -> None:
        """Test performance with many computed columns (benefits from vectorized dispatch)."""
        t = pxt.create_table('wide_tbl', {'c1': pxt.Int, 'c2': pxt.String})

        # Add 20 computed columns to stress the dispatch logic
        for i in range(20):
            t.add_computed_column(**{f'computed_{i}': t.c2 + f'_{i}'})

        row_count = 10000

        def insert_wide_table() -> None:
            t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        benchmark(insert_wide_table)

    @pytest.mark.benchmark(group='dependency_chain')
    def test_dependency_chain_evaluation(self, uses_db: None, benchmark: Any) -> None:
        """Test performance with chained dependencies (tests ready-slot computation)."""
        t = pxt.create_table('chain_tbl', {'text': pxt.String})

        # Create a chain of dependencies: each column depends on the previous
        t.add_computed_column(step1=upper(t.text))
        t.add_computed_column(step2=lower(t.step1))
        t.add_computed_column(step3=upper(t.step2))
        t.add_computed_column(step4=lower(t.step3))
        t.add_computed_column(step5=upper(t.step4))

        row_count = 50000

        def insert_chain() -> None:
            t.insert([{'text': f'Test_String_{i}'} for i in range(row_count)])

        benchmark(insert_chain)

    @pytest.mark.benchmark(group='insert_computed')
    def test_insert_with_computed_columns(self, uses_db: None, benchmark: Any) -> None:
        """Test insert performance when computed columns need evaluation."""

        t = pxt.create_table('insert_tbl', {'c1': pxt.Int, 'c2': pxt.String})
        t.add_computed_column(c3=isascii(t.c2))
        t.add_computed_column(c4=isalpha(t.c2))
        t.add_computed_column(c5=upper(t.c2))
        
        row_count = 50000
        def do_insert() -> None:
            t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        benchmark(do_insert)

    @pytest.mark.benchmark(group='batch_scaling')
    @pytest.mark.parametrize('row_count', [1000, 10000, 50000, 100000])
    def test_batch_scaling(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Test how performance scales with row count (vectorization benefit)."""
        t = pxt.create_table(f'scale_tbl_{row_count}', {'c1': pxt.Int, 'c2': pxt.String})
        t.insert([{'c1': i, 'c2': f'str_{i}'} for i in range(row_count)])

        def select_with_functions() -> None:
            res = t.select(t.c1, isascii(t.c2), isalpha(t.c2)).collect()
            assert len(res) == row_count

        benchmark(select_with_functions)

    # -------------------------------------------------------------------------
    # Image Processing Benchmarks (all go through expr_eval_node)
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='image_properties')
    def test_image_properties(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark image property extraction (width, height, mode)."""
        row_count = 500

        t = pxt.create_table('img_props_tbl', {'img': pxt.Image})
        t.add_computed_column(width=t.img.width)
        t.add_computed_column(height=t.img.height)
        t.add_computed_column(mode=t.img.mode)
        # Duplicate the same image many times

        def insert_images() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_images)

    @pytest.mark.benchmark(group='image_transform')
    def test_image_resize(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark image resize operations."""
        row_count = 200

        t = pxt.create_table('img_resize_tbl', {'img': pxt.Image})
        t.add_computed_column(resized=t.img.resize((128, 128)))

        def insert_resized() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_resized)

    @pytest.mark.benchmark(group='image_transform')
    def test_image_convert_grayscale(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark image color conversion."""
        row_count = 200

        t = pxt.create_table('img_convert_tbl', {'img': pxt.Image})
        t.add_computed_column(grayscale=t.img.convert('L'))

        def insert_converted() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_converted)

    @pytest.mark.benchmark(group='image_chain')
    def test_image_transform_chain(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark chained image transformations (tests dependency resolution)."""
        row_count = 100

        t = pxt.create_table('img_chain_tbl', {'img': pxt.Image})
        # Chain of dependent transformations
        t.add_computed_column(step1=t.img.resize((256, 256)))
        t.add_computed_column(step2=t.step1.convert('L'))
        t.add_computed_column(step3=t.step2.rotate(90))
        t.add_computed_column(final_width=t.step3.width)

        def insert_chain() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_chain)

    @pytest.mark.benchmark(group='image_wide')
    def test_wide_image_table(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark table with many image-derived computed columns."""
        row_count = 100

        t = pxt.create_table('img_wide_tbl', {'img': pxt.Image})
        # Many computed columns from the same source
        t.add_computed_column(width=t.img.width)
        t.add_computed_column(height=t.img.height)
        t.add_computed_column(mode=t.img.mode)
        t.add_computed_column(resized_small=t.img.resize((64, 64)))
        t.add_computed_column(resized_med=t.img.resize((128, 128)))
        t.add_computed_column(grayscale=t.img.convert('L'))
        t.add_computed_column(rotated=t.img.rotate(45))
        t.add_computed_column(thumb=t.img.resize((32, 32)))
        t.add_computed_column(flipped=t.img.transpose(0))  # FLIP_LEFT_RIGHT
        t.add_computed_column(cropped=t.img.crop((10, 10, 100, 100)))

        def insert_wide() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_wide)

    @pytest.mark.benchmark(group='image_histogram')
    def test_image_histogram(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark compute-intensive image histogram calculation."""
        row_count = 100

        t = pxt.create_table('img_hist_tbl', {'img': pxt.Image})
        t.add_computed_column(histogram=t.img.histogram())
        t.add_computed_column(extrema=t.img.getextrema())

        def insert_histogram() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_histogram)

    # -------------------------------------------------------------------------
    # Video Processing Benchmarks
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='video_metadata')
    def test_video_metadata(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark video metadata extraction."""
        video_files = get_video_files()
        if not video_files:
            pytest.skip('No test videos available')
        video_path = video_files[0]

        t = pxt.create_table('video_meta_tbl', {'video': pxt.Video})
        t.add_computed_column(metadata=pxt_video.get_metadata(t.video))
        t.add_computed_column(duration=pxt_video.get_duration(t.video))

        def insert_metadata() -> None:
            t.insert([{'video': video_path} for _ in range(20)])

        benchmark(insert_metadata)

    @pytest.mark.benchmark(group='video_frames')
    def test_video_frame_extraction(self, uses_db: None, benchmark: Any) -> None:
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

    # -------------------------------------------------------------------------
    # Audio Processing Benchmarks
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='audio_metadata')
    def test_audio_metadata(self, uses_db: None, benchmark: Any) -> None:
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

    # -------------------------------------------------------------------------
    # Scaling Benchmarks
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='image_scaling')
    @pytest.mark.parametrize('row_count', [50, 100, 200, 500])
    def test_image_scaling(self, uses_db: None, benchmark: Any, row_count: int) -> None:
        """Test how image processing scales with row count."""
        t = pxt.create_table(f'img_scale_{row_count}', {'img': pxt.Image})
        t.add_computed_column(width=t.img.width)
        t.add_computed_column(height=t.img.height)
        t.add_computed_column(resized=t.img.resize((64, 64)))

        def insert_images() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_images)

    # -------------------------------------------------------------------------
    # Mixed Multimodal Benchmarks
    # -------------------------------------------------------------------------

    @pytest.mark.benchmark(group='multimodal_mixed')
    def test_mixed_image_transforms(self, uses_db: None, benchmark: Any) -> None:
        """Benchmark table with multiple independent image transform branches."""
        row_count = 100

        t = pxt.create_table('mixed_transforms_tbl', {'img': pxt.Image})
        # Branch 1: resize path
        t.add_computed_column(small=t.img.resize((64, 64)))
        t.add_computed_column(small_gray=t.small.convert('L'))
        # Branch 2: grayscale path
        t.add_computed_column(gray=t.img.convert('L'))
        t.add_computed_column(gray_rotated=t.gray.rotate(90))
        # Branch 3: crop path
        t.add_computed_column(cropped=t.img.crop((0, 0, 100, 100)))
        t.add_computed_column(cropped_resized=t.cropped.resize((50, 50)))

        def insert_branches() -> None:
            t.insert([{'img': SAMPLE_IMAGE_URL} for _ in range(row_count)])

        benchmark(insert_branches)

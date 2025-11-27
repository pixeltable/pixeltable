"""
Performance comparison between HNSW (pgvector) and DiskANN (pgvectorscale) indexes.

This script compares index performance using real data:
- Option 1: Extract frames from a video file
- Option 2: Load a Hugging Face image dataset
- Option 3: Use sample image URLs

Metrics compared:
- Index creation time
- Query latency (p50, p95, p99)
- Query throughput

DiskANN Tuning (from pgvectorscale README):
- Index build params: num_neighbors (default 50), search_list_size (default 100)
- storage_layout: 'memory_optimized' (SBQ compression) or 'plain'
- Query params: SET diskann.query_rescore = N (default 50)
- Memory: SET maintenance_work_mem = '2GB' for large datasets
"""

import argparse
import os
import statistics
import time
from pathlib import Path

import sqlalchemy as sql
from tabulate import tabulate

import pixeltable as pxt


def run_video_frame_comparison(
    video_path: str,
    num_queries: int = 50,
    fps: float = 10.0,
    query_rescore: int = 50,
) -> None:
    """
    Compare HNSW and DiskANN using video frame extraction.

    Args:
        video_path: Path to video file
        num_queries: Number of similarity queries to run
        fps: Frames per second to extract (higher = more frames)
        query_rescore: DiskANN query_rescore parameter (higher = more accurate but slower)
    """
    print(f'\n{"=" * 70}')
    print('HNSW vs DiskANN: Video Frame Embedding Comparison')
    print(f'{"=" * 70}')
    print(f'Video: {video_path}')
    print(f'Frame extraction rate: {fps} fps')
    print(f'Queries per index: {num_queries}')
    print(f'DiskANN query_rescore: {query_rescore}')
    print(f'{"=" * 70}\n')

    from pixeltable.functions.huggingface import clip
    from pixeltable.iterators import FrameIterator

    clip_embed = clip.using(model_id='openai/clip-vit-base-patch32')

    # Setup
    pxt.drop_dir('index_perf_test', force=True)
    pxt.create_dir('index_perf_test')

    results: dict[str, dict] = {}

    for index_type in ['hnsw', 'diskann']:
        print(f'\n{"=" * 70}')
        print(f'Testing {index_type.upper()} Index')
        print(f'{"=" * 70}')

        # Show index parameters
        if index_type == 'hnsw':
            print('  HNSW params: m=16, ef_construction=64')
        else:
            print('  DiskANN params: num_neighbors=50, search_list_size=100, storage=memory_optimized')

        # Create video table
        print(f'\n--- Creating video table for {index_type} ---')
        video_tbl = pxt.create_table(
            f'index_perf_test.video_{index_type}',
            {'video': pxt.Video}
        )

        # Create frame view
        print(f'--- Creating frame extraction view (fps={fps}) ---')
        frame_view = pxt.create_view(
            f'index_perf_test.frames_{index_type}',
            video_tbl,
            iterator=FrameIterator.create(video=video_tbl.video, fps=fps)
        )

        # Insert video
        print(f'--- Inserting video and extracting frames ---')
        insert_start = time.monotonic()
        video_tbl.insert([{'video': video_path}])
        insert_duration = time.monotonic() - insert_start

        # Count frames
        frame_count = frame_view.count()
        print(f'    Extracted {frame_count} frames in {insert_duration:.2f}s')

        # Set maintenance_work_mem for better index build performance
        from pixeltable.env import Env
        engine = Env.get()._sa_engine
        with engine.connect() as conn:
            conn.execute(sql.text("SET maintenance_work_mem = '256MB'"))
            conn.commit()

        # Create embedding index
        print(f'--- Creating {index_type.upper()} embedding index on {frame_count} frames ---')
        index_start = time.monotonic()
        try:
            frame_view.add_embedding_index(
                'frame',
                idx_name=f'{index_type}_idx',
                embedding=clip_embed,
                index_type=index_type,  # type: ignore[arg-type]
            )
            index_duration = time.monotonic() - index_start
            index_success = True
            print(f'    Index created in {index_duration:.2f}s ({frame_count / index_duration:.1f} frames/s)')
        except Exception as e:
            print(f'    ERROR: {e}')
            index_duration = 0.0
            index_success = False

        if not index_success:
            results[index_type] = {'success': False}
            continue

        # Set DiskANN query parameters for better accuracy/speed tradeoff
        if index_type == 'diskann':
            with engine.connect() as conn:
                conn.execute(sql.text(f"SET diskann.query_rescore = {query_rescore}"))
                conn.commit()
            print(f'    Set diskann.query_rescore = {query_rescore}')

        # Run similarity queries
        print(f'--- Running {num_queries} similarity queries ---')
        query_times: list[float] = []

        # Use text queries for multimodal search (CLIP supports text-to-image)
        query_texts = [
            'a busy street with cars and traffic',
            'people walking on sidewalk',
            'tall buildings and skyscrapers',
            'traffic lights and road signs',
            'motorcycles and scooters',
            'shop signs and storefronts',
            'pedestrians crossing the street',
            'urban city scene',
            'city landscape with buildings',
            'evening or night time scene',
            'vehicles on the road',
            'street vendors',
            'public transportation',
            'crowded marketplace',
            'modern architecture',
        ]

        for i in range(num_queries):
            query_text = query_texts[i % len(query_texts)]
            query_start = time.monotonic()
            sim = frame_view.frame.similarity(query_text, idx=f'{index_type}_idx')
            _ = frame_view.order_by(sim, asc=False).limit(5).select(frame_view.frame, similarity=sim).collect()
            query_times.append(time.monotonic() - query_start)

            if (i + 1) % 20 == 0:
                avg_so_far = statistics.mean([t * 1000 for t in query_times])
                print(f'    Completed {i + 1}/{num_queries} queries (avg: {avg_so_far:.1f}ms)')

        # Calculate statistics
        query_times_ms = [t * 1000 for t in query_times]
        results[index_type] = {
            'success': True,
            'frame_count': frame_count,
            'insert_duration': insert_duration,
            'index_duration': index_duration,
            'query_count': num_queries,
            'query_total': sum(query_times),
            'query_avg': statistics.mean(query_times_ms),
            'query_p50': statistics.median(query_times_ms),
            'query_p95': sorted(query_times_ms)[int(len(query_times_ms) * 0.95)],
            'query_p99': sorted(query_times_ms)[int(len(query_times_ms) * 0.99)],
            'qps': num_queries / sum(query_times),
        }

        print(f'    Query Statistics:')
        print(f'      Avg: {results[index_type]["query_avg"]:.1f}ms')
        print(f'      P50: {results[index_type]["query_p50"]:.1f}ms')
        print(f'      P95: {results[index_type]["query_p95"]:.1f}ms')
        print(f'      P99: {results[index_type]["query_p99"]:.1f}ms')
        print(f'      QPS: {results[index_type]["qps"]:.1f}')

    # Print comparison
    _print_comparison(results)

    # Cleanup
    print('\n--- Cleanup ---')
    pxt.drop_dir('index_perf_test', force=True)
    print('Done.')


def run_huggingface_comparison(
    dataset_name: str = 'Multimodal-Fatima/COCO_sample',
    num_rows: int | None = None,
    num_queries: int = 50,
) -> None:
    """
    Compare HNSW and DiskANN using a Hugging Face image dataset.

    Args:
        dataset_name: Name of the HuggingFace dataset
        num_rows: Max rows to use (None for all)
        num_queries: Number of queries to run
    """
    print(f'\n{"=" * 70}')
    print('HNSW vs DiskANN: Hugging Face Dataset Comparison')
    print(f'{"=" * 70}')
    print(f'Dataset: {dataset_name}')
    print(f'Max rows: {num_rows or "all"}')
    print(f'Queries: {num_queries}')
    print(f'{"=" * 70}\n')

    import datasets

    from pixeltable.functions.huggingface import clip

    clip_embed = clip.using(model_id='openai/clip-vit-base-patch32')

    # Load dataset
    print('--- Loading Hugging Face dataset ---')
    load_start = time.monotonic()
    try:
        ds = datasets.load_dataset(dataset_name, split='train')
        if num_rows:
            ds = ds.select(range(min(num_rows, len(ds))))
        print(f'    Loaded {len(ds)} rows in {time.monotonic() - load_start:.1f}s')
    except Exception as e:
        print(f'    ERROR loading dataset: {e}')
        return

    # Setup
    pxt.drop_dir('index_perf_test', force=True)
    pxt.create_dir('index_perf_test')

    results: dict[str, dict] = {}

    for index_type in ['hnsw', 'diskann']:
        print(f'\n{"=" * 70}')
        print(f'Testing {index_type.upper()} Index')
        print(f'{"=" * 70}')

        # Import dataset
        print(f'\n--- Importing dataset for {index_type} ---')
        import_start = time.monotonic()
        try:
            tbl = pxt.io.import_huggingface_dataset(
                f'index_perf_test.images_{index_type}',
                ds,
                on_error='ignore',
                media_validation='on_read',
                if_exists='replace',
            )
            import_duration = time.monotonic() - import_start
            row_count = tbl.count()
            print(f'    Imported {row_count} rows in {import_duration:.1f}s')
        except Exception as e:
            print(f'    ERROR: {e}')
            results[index_type] = {'success': False}
            continue

        # Find image column
        image_col = None
        for col_name in ['image', 'img', 'photo']:
            if col_name in [c.name for c in tbl.columns()]:
                image_col = col_name
                break

        if not image_col:
            print('    ERROR: No image column found')
            results[index_type] = {'success': False}
            continue

        # Create embedding index
        print(f'--- Creating {index_type.upper()} embedding index ---')
        index_start = time.monotonic()
        try:
            tbl.add_embedding_index(
                image_col,
                idx_name=f'{index_type}_idx',
                embedding=clip_embed,
                index_type=index_type,  # type: ignore[arg-type]
            )
            index_duration = time.monotonic() - index_start
            print(f'    Index created in {index_duration:.1f}s')
        except Exception as e:
            print(f'    ERROR: {e}')
            results[index_type] = {'success': False}
            continue

        # Run queries
        print(f'--- Running {num_queries} similarity queries ---')
        query_times: list[float] = []
        query_texts = [
            'a dog playing', 'a cat sleeping', 'people eating',
            'cars on street', 'nature landscape', 'indoor scene',
            'animals', 'food on table', 'sports activity', 'cityscape',
        ]

        img_col_ref = getattr(tbl, image_col)
        for i in range(num_queries):
            query_text = query_texts[i % len(query_texts)]
            query_start = time.monotonic()
            sim = img_col_ref.similarity(query_text, idx=f'{index_type}_idx')
            _ = tbl.order_by(sim, asc=False).limit(5).select(img_col_ref, similarity=sim).collect()
            query_times.append(time.monotonic() - query_start)

            if (i + 1) % 10 == 0:
                print(f'    Completed {i + 1}/{num_queries}')

        query_times_ms = [t * 1000 for t in query_times]
        results[index_type] = {
            'success': True,
            'frame_count': row_count,
            'insert_duration': import_duration,
            'index_duration': index_duration,
            'query_count': num_queries,
            'query_total': sum(query_times),
            'query_avg': statistics.mean(query_times_ms),
            'query_p50': statistics.median(query_times_ms),
            'query_p95': sorted(query_times_ms)[int(len(query_times_ms) * 0.95)],
            'query_p99': sorted(query_times_ms)[int(len(query_times_ms) * 0.99)],
            'qps': num_queries / sum(query_times),
        }

        print(f'    Avg: {results[index_type]["query_avg"]:.1f}ms, QPS: {results[index_type]["qps"]:.1f}')

    _print_comparison(results)

    print('\n--- Cleanup ---')
    pxt.drop_dir('index_perf_test', force=True)


def run_text_comparison(num_rows: int = 2000, num_queries: int = 100) -> None:
    """
    Compare HNSW and DiskANN using generated text data.
    Faster to run, good for quick comparisons.
    """
    print(f'\n{"=" * 70}')
    print('HNSW vs DiskANN: Text Embedding Comparison')
    print(f'{"=" * 70}')
    print(f'Rows: {num_rows:,}')
    print(f'Queries: {num_queries}')
    print(f'{"=" * 70}\n')

    from pixeltable.functions.huggingface import sentence_transformer

    embed_fn = sentence_transformer.using(model_id='sentence-transformers/all-MiniLM-L12-v2')

    # Generate data
    sentences = [
        f'Document {i}: This discusses topic {i % 50} with perspective {i % 7} and details about item {i % 23}.'
        for i in range(num_rows)
    ]
    query_sentences = [f'information about topic {i % 50}' for i in range(num_queries)]

    pxt.drop_dir('index_perf_test', force=True)
    pxt.create_dir('index_perf_test')

    results: dict[str, dict] = {}

    for index_type in ['hnsw', 'diskann']:
        print(f'\n{"=" * 70}')
        print(f'Testing {index_type.upper()} Index')
        print(f'{"=" * 70}')

        # Create and populate table
        print(f'\n--- Creating table and inserting {num_rows:,} rows ---')
        tbl = pxt.create_table(f'index_perf_test.text_{index_type}', {'text': pxt.String})

        insert_start = time.monotonic()
        tbl.insert({'text': s} for s in sentences)
        insert_duration = time.monotonic() - insert_start
        print(f'    Insert: {insert_duration:.1f}s ({num_rows / insert_duration:.0f} rows/s)')

        # Set maintenance_work_mem for better index build performance on large datasets
        if num_rows >= 10000:
            from pixeltable.env import Env
            engine = Env.get()._sa_engine
            work_mem = '512MB' if num_rows >= 50000 else '256MB'
            with engine.connect() as conn:
                conn.execute(sql.text(f"SET maintenance_work_mem = '{work_mem}'"))
                conn.commit()
            print(f'    Set maintenance_work_mem = {work_mem}')

        # Create index
        print(f'--- Creating {index_type.upper()} index ---')
        index_start = time.monotonic()
        try:
            tbl.add_embedding_index('text', idx_name=f'{index_type}_idx', embedding=embed_fn, index_type=index_type)  # type: ignore[arg-type]
            index_duration = time.monotonic() - index_start
            print(f'    Index: {index_duration:.1f}s ({num_rows / index_duration:.0f} rows/s)')
        except Exception as e:
            print(f'    ERROR: {e}')
            results[index_type] = {'success': False}
            continue

        # Run queries
        print(f'--- Running {num_queries} queries ---')
        query_times: list[float] = []

        for i, q in enumerate(query_sentences):
            start = time.monotonic()
            sim = tbl.text.similarity(q, idx=f'{index_type}_idx')
            _ = tbl.order_by(sim, asc=False).limit(10).select(tbl.text, similarity=sim).collect()
            query_times.append(time.monotonic() - start)

            if (i + 1) % 25 == 0:
                print(f'    {i + 1}/{num_queries} queries done')

        query_times_ms = [t * 1000 for t in query_times]
        results[index_type] = {
            'success': True,
            'frame_count': num_rows,
            'insert_duration': insert_duration,
            'index_duration': index_duration,
            'query_count': num_queries,
            'query_total': sum(query_times),
            'query_avg': statistics.mean(query_times_ms),
            'query_p50': statistics.median(query_times_ms),
            'query_p95': sorted(query_times_ms)[int(len(query_times_ms) * 0.95)],
            'query_p99': sorted(query_times_ms)[int(len(query_times_ms) * 0.99)],
            'qps': num_queries / sum(query_times),
        }

    _print_comparison(results)

    print('\n--- Cleanup ---')
    pxt.drop_dir('index_perf_test', force=True)


def _print_comparison(results: dict[str, dict]) -> None:
    """Print comparison summary table."""
    print(f'\n{"=" * 70}')
    print('COMPARISON SUMMARY')
    print(f'{"=" * 70}\n')

    hnsw = results.get('hnsw', {})
    diskann = results.get('diskann', {})

    if not (hnsw.get('success') and diskann.get('success')):
        print('One or more index types failed.')
        return

    def pct_diff(a: float, b: float) -> str:
        if a == 0:
            return 'N/A'
        diff = ((b - a) / a) * 100
        return f'{diff:+.1f}%'

    headers = ['Metric', 'HNSW', 'DiskANN', 'Diff', 'Winner']
    data = []

    # Data size
    data.append(['Data Points', f'{hnsw["frame_count"]:,}', f'{diskann["frame_count"]:,}', '-', '-'])

    # Index creation
    h_idx, d_idx = hnsw['index_duration'], diskann['index_duration']
    winner = '🏆 DiskANN' if d_idx < h_idx else '🏆 HNSW'
    data.append(['Index Creation', f'{h_idx:.1f}s', f'{d_idx:.1f}s', pct_diff(h_idx, d_idx), winner])

    # Indexing speed
    h_spd = hnsw['frame_count'] / h_idx if h_idx > 0 else 0
    d_spd = diskann['frame_count'] / d_idx if d_idx > 0 else 0
    winner = '🏆 DiskANN' if d_spd > h_spd else '🏆 HNSW'
    data.append(['Index Speed', f'{h_spd:.0f}/s', f'{d_spd:.0f}/s', pct_diff(h_spd, d_spd), winner])

    # Query latency (avg)
    h_avg, d_avg = hnsw['query_avg'], diskann['query_avg']
    winner = '🏆 DiskANN' if d_avg < h_avg else '🏆 HNSW'
    data.append(['Avg Latency', f'{h_avg:.1f}ms', f'{d_avg:.1f}ms', pct_diff(h_avg, d_avg), winner])

    # Query latency (p50)
    h_p50, d_p50 = hnsw['query_p50'], diskann['query_p50']
    winner = '🏆 DiskANN' if d_p50 < h_p50 else '🏆 HNSW'
    data.append(['P50 Latency', f'{h_p50:.1f}ms', f'{d_p50:.1f}ms', pct_diff(h_p50, d_p50), winner])

    # Query latency (p95)
    h_p95, d_p95 = hnsw['query_p95'], diskann['query_p95']
    winner = '🏆 DiskANN' if d_p95 < h_p95 else '🏆 HNSW'
    data.append(['P95 Latency', f'{h_p95:.1f}ms', f'{d_p95:.1f}ms', pct_diff(h_p95, d_p95), winner])

    # Query latency (p99)
    h_p99, d_p99 = hnsw['query_p99'], diskann['query_p99']
    winner = '🏆 DiskANN' if d_p99 < h_p99 else '🏆 HNSW'
    data.append(['P99 Latency', f'{h_p99:.1f}ms', f'{d_p99:.1f}ms', pct_diff(h_p99, d_p99), winner])

    # Throughput
    h_qps, d_qps = hnsw['qps'], diskann['qps']
    winner = '🏆 DiskANN' if d_qps > h_qps else '🏆 HNSW'
    data.append(['Throughput', f'{h_qps:.1f} q/s', f'{d_qps:.1f} q/s', pct_diff(h_qps, d_qps), winner])

    print(tabulate(data, headers=headers, tablefmt='grid'))

    # Overall winner
    diskann_wins = sum(1 for row in data[1:] if 'DiskANN' in row[-1])
    hnsw_wins = sum(1 for row in data[1:] if 'HNSW' in row[-1])

    print(f'\n📊 Overall: DiskANN wins {diskann_wins}/{len(data) - 1} metrics, HNSW wins {hnsw_wins}/{len(data) - 1} metrics')

    if diskann_wins > hnsw_wins:
        print('🎉 DiskANN (pgvectorscale) is the overall winner!')
    elif hnsw_wins > diskann_wins:
        print('🎉 HNSW (pgvector) is the overall winner!')
    else:
        print('🤝 It\'s a tie!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare HNSW and DiskANN index performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick text comparison (fastest)
  python index_comparison_profiler.py --mode text --rows 2000 --queries 100

  # Video frame comparison
  python index_comparison_profiler.py --mode video --video docs/resources/bangkok.mp4 --queries 50

  # Hugging Face dataset comparison
  python index_comparison_profiler.py --mode huggingface --dataset Multimodal-Fatima/COCO_sample --rows 500
        """,
    )
    parser.add_argument(
        '--mode',
        choices=['text', 'video', 'huggingface'],
        default='text',
        help='Comparison mode (default: text)',
    )
    parser.add_argument('--rows', type=int, default=2000, help='Number of rows for text mode (default: 2000)')
    parser.add_argument('--queries', type=int, default=100, help='Number of queries (default: 100)')
    parser.add_argument('--video', type=str, default='docs/resources/bangkok.mp4', help='Video path for video mode')
    parser.add_argument('--fps', type=float, default=10.0, help='FPS for video frame extraction (default: 10.0)')
    parser.add_argument('--rescore', type=int, default=50, help='DiskANN query_rescore param (default: 50, higher=more accurate)')
    parser.add_argument(
        '--dataset',
        type=str,
        default='Multimodal-Fatima/COCO_sample',
        help='HuggingFace dataset name',
    )

    args = parser.parse_args()

    os.environ['PIXELTABLE_DB'] = 'test_index_comparison'
    pxt.init()

    if args.mode == 'text':
        run_text_comparison(num_rows=args.rows, num_queries=args.queries)
    elif args.mode == 'video':
        run_video_frame_comparison(video_path=args.video, num_queries=args.queries, fps=args.fps, query_rescore=args.rescore)
    elif args.mode == 'huggingface':
        run_huggingface_comparison(dataset_name=args.dataset, num_rows=args.rows, num_queries=args.queries)

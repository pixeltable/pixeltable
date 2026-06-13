"""OTEL demo: exercises the new instrumentation and exports traces to a local Phoenix server.

Run from the repo root with a Phoenix server running (`phoenix serve`):

    PIXELTABLE_HOME=/tmp/pxt_otel_demo python otel_demo.py

With pixeltable[otel] installed and no OTEL_* env vars set, instrumentation is on by default and
traces flow to Phoenix at http://localhost:6006 (project 'pixeltable').

Workloads:
1. (span_level=TRACE) video inserts: frame extraction view + image resize + HF CLIP embeddings per
   frame; TRACE adds one span per iterator step and per resize call
2. (span_level=DEBUG) HF sentence-transformer embeddings over a text table (batched UDF: one span per
   32-row batch, which is one actual model invocation)
3. (span_level=INFO) large scalar insert (50k rows with a computed scalar UDF column), showing
   aggregated per-UDF stats on the exec.ExprEvalNode span instead of 50k call spans
4. (span_level=TRACE) small scalar insert (500 rows): one span per individual UDF invocation, in
   addition to the same aggregates
"""

import glob
import time

import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable import hooks
from pixeltable.functions.huggingface import clip, sentence_transformer

# UDFs must live in an importable module (not a script's global namespace)
from tests.test_hooks import _double


def main() -> None:
    pxt.init()  # default-on otel initializes here (Phoenix register, project 'pixeltable')

    pxt.create_dir('otel_demo', if_exists='ignore')

    # phase 1 at TRACE: per-batch/per-call udf spans plus per-row and per-iterator-step spans
    hooks.set_span_level(hooks.TRACE)

    print('=== phase 1: video frame extraction + CLIP embeddings (TRACE) ===')
    t0 = time.perf_counter()
    videos = pxt.create_table('otel_demo.videos', {'video': pxt.Video}, if_exists='replace_force')
    videos.add_computed_column(metadata=pxtf.video.get_metadata(videos.video))
    frames = pxt.create_view(
        'otel_demo.frames', videos, iterator=pxtf.video.frame_iterator(videos.video, fps=1), if_exists='replace_force'
    )
    frames.add_computed_column(small=frames.frame.resize((224, 224)))
    frames.add_computed_column(clip_embed=clip(frames.small, model_id='openai/clip-vit-base-patch32'))

    video_paths = sorted(p for p in glob.glob('tests/data/videos/*') if 'bad_video' not in p)[:3]
    assert video_paths, 'run from the repo root'
    videos.insert({'video': p} for p in video_paths)
    print(f'    {frames.count()} frames embedded in {time.perf_counter() - t0:.1f}s')

    hooks.set_span_level(hooks.DEBUG)
    print('=== phase 2: sentence-transformer embeddings (batched HF udf, DEBUG) ===')
    t0 = time.perf_counter()
    docs = pxt.create_table('otel_demo.docs', {'text': pxt.String}, if_exists='replace_force')
    docs.add_computed_column(
        embed=sentence_transformer(docs.text, model_id='sentence-transformers/all-MiniLM-L6-v2')
    )
    docs.insert({'text': f'pixeltable demo sentence number {i}, about observability'} for i in range(200))
    print(f'    200 docs embedded in {time.perf_counter() - t0:.1f}s')

    # phase 3 at INFO: no per-call spans; the ExprEvalNode span carries pxt.udf.* aggregates
    hooks.set_span_level(hooks.INFO)

    print('=== phase 3: large scalar insert (50k rows, computed scalar udf, INFO) ===')
    t0 = time.perf_counter()
    big = pxt.create_table('otel_demo.big', {'c1': pxt.Int, 'c2': pxt.String}, if_exists='replace_force')
    big.add_computed_column(doubled=_double(big.c1))
    big.insert({'c1': i, 'c2': f'str_{i}'} for i in range(50_000))
    print(f'    50k rows inserted in {time.perf_counter() - t0:.1f}s')

    # most verbose mode: every individual UDF invocation gets its own span under exec.ExprEvalNode,
    # alongside the same pxt.udf.* aggregates on the node span
    hooks.set_span_level(hooks.TRACE)

    print('=== phase 4: small scalar insert (500 rows, one span per udf call, TRACE) ===')
    t0 = time.perf_counter()
    small = pxt.create_table('otel_demo.small', {'c1': pxt.Int, 'c2': pxt.String}, if_exists='replace_force')
    small.add_computed_column(doubled=_double(small.c1))
    small.insert({'c1': i, 'c2': f'str_{i}'} for i in range(500))
    print(f'    500 rows inserted in {time.perf_counter() - t0:.1f}s')

    # give the BatchSpanProcessor a moment, then flush whatever is left
    import os

    import pixeltable.otel._sdk as _sdk

    if _sdk._state.tracer_provider is not None:
        _sdk._state.tracer_provider.force_flush()
    backend = os.environ.get('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://localhost:6006 (Phoenix default)')
    print(f'traces exported to {backend}')


if __name__ == '__main__':
    main()

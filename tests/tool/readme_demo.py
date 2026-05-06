"""
README Quick Start demo — runs the full example end-to-end.

Usage:
    # Requires GEMINI_API_KEY in env or ~/.pixeltable/config.toml
    python tests/tool/readme_demo.py

    # Or pass it explicitly:
    GEMINI_API_KEY=... python tests/tool/readme_demo.py

Dependencies:
    pip install pixeltable google-genai torch transformers scenedetect 'fastapi[standard]'
"""

import sys
import textwrap
import time

import pixeltable as pxt
from pixeltable.functions import gemini, huggingface

NAMESPACE = 'readme_demo'
TABLE = f'{NAMESPACE}/video_search'
BASE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources'


def step(msg: str) -> None:
    print(f'\n{"=" * 60}\n  {msg}\n{"=" * 60}')


def main() -> None:
    # ------------------------------------------------------------------
    step('1. Setup: create table')
    # ------------------------------------------------------------------
    pxt.drop_dir(NAMESPACE, force=True, if_not_exists='ignore')
    pxt.create_dir(NAMESPACE, if_exists='ignore')

    videos = pxt.create_table(TABLE, {'video': pxt.Video, 'title': pxt.String}, if_exists='replace')
    print(f'  Table created: {TABLE}')

    # ------------------------------------------------------------------
    step('2. Orchestrate: add computed columns')
    # ------------------------------------------------------------------
    videos.add_computed_column(scenes=videos.video.scene_detect_adaptive(), if_exists='ignore')
    print('  + scenes (scene_detect_adaptive)')

    videos.add_computed_column(
        response=gemini.generate_content(
            [videos.video, 'Describe this video in detail.'], model='gemini-2.5-flash'
        ),
        if_exists='ignore',
    )
    print('  + response (gemini.generate_content)')

    videos.add_computed_column(
        description=videos.response.candidates[0].content.parts[0].text,
        if_exists='ignore',
    )
    print('  + description (JSON path extraction)')

    # ------------------------------------------------------------------
    step('3. Index: add embedding indexes')
    # ------------------------------------------------------------------
    videos.add_embedding_index(
        'video', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'), if_not_exists=True
    )
    print('  + video embedding index')

    videos.add_embedding_index(
        'description', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'), if_not_exists=True
    )
    print('  + description embedding index')

    # ------------------------------------------------------------------
    step('4. Insert: trigger the full pipeline')
    # ------------------------------------------------------------------
    t0 = time.time()
    status = videos.insert([
        {'video': f'{BASE_URL}/bangkok.mp4', 'title': 'Bangkok Street Tour'},
        {'video': f'{BASE_URL}/The-Pursuit-of-Happiness-Video-Extract.mp4', 'title': 'The Pursuit of Happiness'},
    ])
    elapsed = time.time() - t0
    print(f'  Inserted {status.num_rows} rows in {elapsed:.1f}s')
    print(f'  Errors: {status.num_excs}')

    # ------------------------------------------------------------------
    step('5. Retrieve: select with on-the-fly transforms')
    # ------------------------------------------------------------------
    results = videos.select(
        videos.video,
        videos.title,
        videos.description,
        detections=huggingface.detr_for_object_detection(
            videos.video.extract_frame(timestamp=2.0),
            model_id='facebook/detr-resnet-50',
        ),
    ).collect()

    print(f'  Got {len(results)} rows')
    for i, row in enumerate(results):
        desc_preview = (row['description'] or '')[:80]
        n_det = len(row['detections'].get('labels', [])) if row['detections'] else 0
        print(f'  [{i}] {row["title"]}: {n_det} detections, desc="{desc_preview}..."')

    # ------------------------------------------------------------------
    step('6. Cross-modal search: find videos by reference image')
    # ------------------------------------------------------------------
    sim = videos.video.similarity(image=f'{BASE_URL}/The-Pursuit-of-Happiness-Screenshot.png')
    search_results = (
        videos.where(videos.description != None)
        .order_by(sim, asc=False)
        .limit(5)
        .select(videos.title, videos.description, similarity=sim)
        .collect()
    )
    print(f'  Cross-modal results ({len(search_results)} rows):')
    for row in search_results:
        print(f'    {row["title"]}: similarity={row["similarity"]:.4f}')

    # ------------------------------------------------------------------
    step('7. Serve: define @pxt.query and test with FastAPIRouter')
    # ------------------------------------------------------------------
    @pxt.query
    def search_videos(query_text: str, limit: int = 5):
        sim = videos.description.similarity(string=query_text)
        return videos.order_by(sim, asc=False).limit(limit).select(videos.title, videos.description, sim)

    # Test the query directly
    direct = search_videos('street food').collect()
    print(f'  Direct query returned {len(direct)} rows')
    for row in direct:
        print(f'    {row["title"]}')

    # Test via FastAPIRouter + TestClient
    import fastapi
    from fastapi.testclient import TestClient
    from pixeltable.serving import FastAPIRouter

    app = fastapi.FastAPI()
    router = FastAPIRouter()

    router.add_query_route(path='/search', query=search_videos)
    router.add_insert_route(
        videos, path='/ingest', inputs=['video', 'title'], outputs=['title', 'description']
    )

    app.include_router(router)
    client = TestClient(app)

    # Query route
    resp = client.post('/search', json={'query_text': 'street food', 'limit': 2})
    assert resp.status_code == 200, f'Query route failed: {resp.status_code} {resp.text}'
    body = resp.json()
    print(f'\n  POST /search -> {len(body["rows"])} rows')
    for row in body['rows']:
        print(f'    {row["title"]}')

    # Insert route
    resp = client.post('/ingest', json={
        'video': f'{BASE_URL}/bangkok.mp4',
        'title': 'Bangkok Street Tour (duplicate)',
    })
    assert resp.status_code == 200, f'Insert route failed: {resp.status_code} {resp.text}'
    ingest_body = resp.json()
    print(f'\n  POST /ingest -> title={ingest_body.get("title")}, desc={str(ingest_body.get("description", ""))[:60]}...')

    # ------------------------------------------------------------------
    step('8. Summary')
    # ------------------------------------------------------------------
    row_count = videos.count()
    cols = [c.name for c in videos.columns()]
    print(f'  Table: {TABLE}')
    print(f'  Rows: {row_count}')
    print(f'  Columns: {", ".join(cols)}')
    print(textwrap.dedent('''
        Full pipeline verified:
          [x] Storage       — create_table, insert
          [x] Orchestrate   — computed columns (scene detect, Gemini, JSON path)
          [x] Index         — embedding indexes (video + text)
          [x] Retrieve      — select with transforms, cross-modal similarity
          [x] Serve         — @pxt.query, FastAPIRouter (query + insert routes)
    '''))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'\nFATAL: {e}', file=sys.stderr)
        raise SystemExit(1)

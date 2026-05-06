"""
Pixeltable end-to-end demo: Storage, Orchestration, Retrieval, Serving

    GEMINI_API_KEY=... python tests/tool/readme_demo.py

    pip install pixeltable google-genai 'fastapi[standard]'
    pip install torch transformers  # optional, for object detection step
"""

import pixeltable as pxt
from pixeltable.functions import gemini

BASE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources'

# ── Storage ──────────────────────────────────────────────────────────
print('\n── Storage ─────────────────────────────────────────')
pxt.drop_dir('demo', force=True, if_not_exists='ignore')
pxt.create_dir('demo')
videos = pxt.create_table('demo/videos', {'video': pxt.Video, 'title': pxt.String})
print(videos)

# ── Orchestration ────────────────────────────────────────────────────
print('\n── Orchestration ───────────────────────────────────')
videos.add_computed_column(
    response=gemini.generate_content(
        [videos.video, 'Describe this video in detail.'], model='gemini-3-flash-preview'
    )
)
videos.add_computed_column(
    description=videos.response.candidates[0].content.parts[0].text.astype(pxt.String)
)
print('Computed columns: response (Gemini), description (extracted text)')

# ── Indexing ─────────────────────────────────────────────────────────
print('\n── Indexing ────────────────────────────────────────')
videos.add_embedding_index('description', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'))
print('Embedding index on description column')

# ── Insert (triggers the full pipeline) ──────────────────────────────
print('\n── Insert (triggers full pipeline) ─────────────────')
videos.insert([
    {'video': f'{BASE_URL}/bangkok.mp4', 'title': 'Bangkok Street Tour'},
    {'video': f'{BASE_URL}/The-Pursuit-of-Happiness-Video-Extract.mp4', 'title': 'The Pursuit of Happiness'},
])
print(videos.select(videos.title, videos.description).collect())

videos = pxt.get_table('demo/videos')

# ── Retrieval: object detection (optional) ───────────────────────────
try:
    from pixeltable.functions import huggingface
    print('\n── Retrieval: on-the-fly object detection ──────────')
    result = videos.select(
        videos.title,
        detections=huggingface.detr_for_object_detection(
            videos.video.extract_frame(timestamp=2.0), model_id='facebook/detr-resnet-50',
        ),
    ).collect()
    print(result)
except Exception as e:
    print(f'\n── Skipping object detection (missing deps: {type(e).__name__}) ──')

# ── Retrieval: semantic search ───────────────────────────────────────
print('\n── Retrieval: semantic search ──────────────────────')
sim = videos.description.similarity(string='street food')
result = videos.order_by(sim, asc=False).limit(5).select(videos.title, sim).collect()
print(result)

# ── Serving ──────────────────────────────────────────────────────────
print('\n── Serving ─────────────────────────────────────────')

@pxt.query
def search_videos(query_text: str, limit: int = 5):
    sim = videos.description.similarity(string=query_text)
    return videos.order_by(sim, asc=False).limit(limit).select(videos.title, videos.description, sim)

import json
import fastapi
from fastapi.testclient import TestClient
from pixeltable.serving import FastAPIRouter

app = fastapi.FastAPI()
router = FastAPIRouter()
router.add_query_route(path='/search', query=search_videos)
router.add_insert_route(videos, path='/ingest', inputs=['video', 'title'], outputs=['title', 'description'])
app.include_router(router)

client = TestClient(app)

print('\nPOST /search {"query_text": "street food", "limit": 2}')
resp = client.post('/search', json={'query_text': 'street food', 'limit': 2})
print(json.dumps(resp.json(), indent=2))

print('\nPOST /ingest (insert a new video via HTTP)')
resp = client.post('/ingest', json={'video': f'{BASE_URL}/bangkok.mp4', 'title': 'Test Upload'})
print(json.dumps(resp.json(), indent=2))

print('\n── Done ────────────────────────────────────────────')

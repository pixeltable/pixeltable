"""
Pixeltable end-to-end demo: Storage → Orchestration → Retrieval → Serving

    GEMINI_API_KEY=... python tests/tool/readme_demo.py

    pip install pixeltable google-genai torch transformers scenedetect 'fastapi[standard]'
"""

import pixeltable as pxt
from pixeltable.functions import gemini, huggingface

BASE_URL = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources'

# ── Storage ──────────────────────────────────────────────────────────
pxt.drop_dir('demo', force=True, if_not_exists='ignore')
pxt.create_dir('demo')

videos = pxt.create_table('demo/videos', {'video': pxt.Video, 'title': pxt.String})

# ── Orchestration ────────────────────────────────────────────────────
videos.add_computed_column(scenes=videos.video.scene_detect_adaptive())

videos.add_computed_column(
    response=gemini.generate_content(
        [videos.video, 'Describe this video in detail.'], model='gemini-3-flash'
    )
)

videos.add_computed_column(description=videos.response.candidates[0].content.parts[0].text)

# ── Indexing ─────────────────────────────────────────────────────────
videos.add_embedding_index('video', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'))
videos.add_embedding_index('description', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'))

# ── Insert (triggers the full pipeline) ──────────────────────────────
videos.insert([
    {'video': f'{BASE_URL}/bangkok.mp4', 'title': 'Bangkok Street Tour'},
    {'video': f'{BASE_URL}/The-Pursuit-of-Happiness-Video-Extract.mp4', 'title': 'The Pursuit of Happiness'},
])

# ── Retrieval: select with on-the-fly object detection ───────────────
videos.select(
    videos.title,
    videos.description,
    detections=huggingface.detr_for_object_detection(
        videos.video.extract_frame(timestamp=2.0), model_id='facebook/detr-resnet-50',
    ),
).collect()

# ── Retrieval: cross-modal search (find video by image) ──────────────
sim = videos.video.similarity(image=f'{BASE_URL}/The-Pursuit-of-Happiness-Screenshot.png')
videos.where(videos.description != None).order_by(sim, asc=False).limit(5).collect()

# ── Serving ──────────────────────────────────────────────────────────
@pxt.query
def search_videos(query_text: str, limit: int = 5):
    sim = videos.description.similarity(string=query_text)
    return videos.order_by(sim, asc=False).limit(limit).select(videos.title, videos.description, sim)

import fastapi
from fastapi.testclient import TestClient
from pixeltable.serving import FastAPIRouter

app = fastapi.FastAPI()
router = FastAPIRouter()
router.add_query_route(path='/search', query=search_videos)
router.add_insert_route(videos, path='/ingest', inputs=['video', 'title'], outputs=['title', 'description'])
app.include_router(router)

client = TestClient(app)
client.post('/search', json={'query_text': 'street food', 'limit': 2}).json()
client.post('/ingest', json={'video': f'{BASE_URL}/bangkok.mp4', 'title': 'Test Upload'}).json()

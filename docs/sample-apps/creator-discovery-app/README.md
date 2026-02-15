# Creator Discovery App

<a href="https://github.com/pixeltable/pixeltable"><img src="https://img.shields.io/badge/Powered%20by-Pixeltable-blue.svg"/></a>

AI-powered creator-brand matching platform built on [Pixeltable](https://github.com/pixeltable/pixeltable) and [Twelve Labs](https://twelvelabs.io/). Pixeltable replaces Pinecone for vector storage and orchestrates Twelve Labs' multimodal embeddings.

## Features

| Feature | Pixeltable Concepts |
|---------|---------------------|
| **Creator-Brand Match** — find matching videos across libraries | `add_embedding_index`, `.similarity()`, `video_splitter` |
| **Semantic Search** — text & image search across video content | Cross-modal embeddings, text-to-video search |
| **Brand Mention Detection** — detect brands in creator videos | `frame_iterator`, computed columns, `openai.vision` |

## How It Works — Data Flow

### Feature 1: Creator-Brand Match

```
INSERT video ──► Pixeltable auto-pipeline:
                   │
                   ├─► store video file
                   ├─► video_splitter (5s chunks)
                   ├─► Twelve Labs embed API (title → 512-dim vector)
                   └─► pgvector index (automatic)

QUERY:
  sim = target.title.similarity(string=source_title)   ← one line
  target.order_by(sim).limit(5).collect()               replaces
                                                        Embed API +
                                                        Pinecone query
```

### Feature 2: Semantic Search

```
TEXT SEARCH:
  sim = table.title.similarity(string="luxury travel")
  table.order_by(sim, asc=False).limit(10).collect()

IMAGE SEARCH (cross-modal — same marengo3.0 embedding space):
  sim = segments.video_segment.similarity(image=pil_image)
  segments.order_by(sim, asc=False).limit(10).collect()
```

### Feature 3: Brand Mention Detection

```
INSERT video ──► Pixeltable auto-pipeline:
                   │
                   ├─► frame_iterator (1 frame / 5s)
                   ├─► openai.vision(prompt, frame)       ← computed column
                   └─► parse_brand_mentions(response)      ← @pxt.udf

API endpoint just READS pre-computed results:
  fv.where(fv.title == title).select(fv.pos, fv.brand_analysis).collect()
  └─► zero AI work at query time — everything was computed on insert
```

## Project Structure

```
config.py              # Settings, model IDs, sample data (YouTube URLs)
functions.py           # Custom @pxt.udf (parse_brand_mentions, gemini_analyze_frame)
setup_pixeltable.py    # Schema: tables, views, indexes, computed columns (idempotent)
app.py                 # FastAPI routes — thin read layer over Pixeltable
frontend/              # Next.js 15 + TypeScript + Tailwind CSS
```

## Prerequisites

- Python 3.10+, Node.js 18+
- [Twelve Labs API Key](https://playground.twelvelabs.io/) (required)
- OpenAI or Gemini API Key (optional, for brand mention detection)

## Setup

### Backend

```bash
cd docs/sample-apps/creator-discovery-app
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip uninstall uvloop -y   # incompatible with Pixeltable's nest_asyncio

# Set API keys in config.py or environment
export TWELVELABS_API_KEY=your_key
export OPENAI_API_KEY=your_key   # optional

# Start — auto-seeds YouTube sample videos on first run (~2 min)
python app.py
```

### Frontend

```bash
cd frontend && npm install && npm run dev
```

### Verify

- API docs: http://localhost:8000/docs
- Frontend: http://localhost:3000

## Important: Event Loop

Always run `python app.py` (single process). Do **not** use `fastapi dev` — its `--reload` spawns a child process that breaks Pixeltable's `nest_asyncio` event loop patching.

## Vision Provider

Brand mention detection defaults to OpenAI Vision. Switch to Gemini:

```bash
export VISION_PROVIDER=gemini
export GEMINI_API_KEY=your_key
```

## Tech Stack

**Backend:** Python, FastAPI, Pixeltable, Twelve Labs, OpenAI/Gemini
**Frontend:** Next.js 15, TypeScript, Tailwind CSS
**Vector Search:** Pixeltable built-in (replaces Pinecone)
**Video Download:** pytubefix (YouTube)
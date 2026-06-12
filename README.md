<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c5ab123e-806c-49bf-93e7-151353719b16">
  <img alt="Pixeltable Logo" src="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131" width="40%">
</picture>

<div>
<br>
</div>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)

[**Quick Start**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**CLI**](https://docs.pixeltable.com/platform/cli) |
[**Dashboard**](https://docs.pixeltable.com/platform/dashboard) |
[**llms-full.txt**](https://docs.pixeltable.com/llms-full.txt) |
[**Starter Kit**](https://github.com/pixeltable/pixeltable-starter-kit) |
[**AI Coding Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**Discord**](https://discord.gg/QPyqFYx2UN)

## Schema-Defined Backend for Multimodal AI Apps

**Define your backend in Python schema — tables, transforms, indexes, and APIs in one place.** Chunking, embeddings, agents, and serving run from computed columns on insert, not edge functions and glue scripts you maintain separately.

Use it as your AI backend alongside any auth/frontend stack, or ship a full app with FastAPI + React via the [Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit).

## Core Capabilities

Expand any row for what Pixeltable replaces, a quick example, and doc links. Examples assume `import pixeltable as pxt`.

<details>
<summary><b>Store:</b> unified multimodal interface</summary>
<br>

[`pxt.Image`](https://docs.pixeltable.com/platform/type-system), `pxt.Video`, `pxt.Audio`, `pxt.Document`, `pxt.Json`: one table for structured and media data with `destination=` for S3, GCS, Azure, R2, and more. Not S3 + Postgres + boto3 sync.

```python
t = pxt.create_table(
    'media',
    {
        'img': pxt.Image,
        'video': pxt.Video,
        'audio': pxt.Audio,
        'document': pxt.Document,
        'metadata': pxt.Json,
    },
)
```

[Type system](https://docs.pixeltable.com/platform/type-system) · [Tables & data](https://docs.pixeltable.com/tutorials/tables-and-data-operations) · [Cloud storage](https://docs.pixeltable.com/integrations/cloud-storage)
</details>

<details>
<summary><b>Import / export:</b> I/O without glue scripts</summary>
<br>

`import_csv()`, Hugging Face, `export_parquet()`, PyTorch, COCO, and more. Not per-format ETL scripts.

```python
# Import from files, URLs, S3, Hugging Face
t.insert(pxt.io.import_csv('data.csv'))
t.insert(pxt.io.import_huggingface_dataset(dataset))

# Export to analytics/ML formats
pxt.io.export_parquet(t, 'data.parquet')
pytorch_ds = t.to_pytorch_dataset('pt')  # PyTorch DataLoader ready
coco_path = t.to_coco_dataset()  # COCO annotations
```

[CSV import](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [Hugging Face](https://docs.pixeltable.com/howto/cookbooks/data/data-import-huggingface) · [PyTorch export](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch) · [ML data wrangling](https://docs.pixeltable.com/use-cases/ml-data-wrangling)
</details>

<details>
<summary><b>Iterate:</b> explode media into rows</summary>
<br>

`create_view()` with iterators splits documents into chunks, video into frames, audio into segments, and typed JSON lists into rows. Not FFmpeg/spaCy pipelines with child tables and foreign keys. For custom explode logic, use [`@pxt.iterator`](https://docs.pixeltable.com/platform/iterators#custom-iterators-with-pxtiterator).

```python
from pixeltable.functions.document import document_splitter
from pixeltable.functions.json import list_iterator
from pixeltable.functions.video import frame_iterator

# Document chunking with overlap
chunks = pxt.create_view(
    'chunks',
    docs,
    iterator=document_splitter(
        document=docs.doc,
        separators='sentence,token_limit',
        overlap=50,
        limit=500,
    ),
)

# Video frame extraction
frames = pxt.create_view(
    'frames',
    videos,
    iterator=frame_iterator(video=videos.video, fps=0.5),
)

# JSON list column: one row per element (typed pxt.Json column required)
items = pxt.create_view('items', t, iterator=list_iterator(t.tags))
```

[Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [Custom iterators](https://docs.pixeltable.com/howto/cookbooks/core/custom-iterators) · [RAG pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline)
</details>

<details>
<summary><b>Orchestrate:</b> declarative computed columns</summary>
<br>

`add_computed_column()` runs incrementally on new or stale rows only. Built-ins cover media processing, embeddings, and [30+ providers](https://docs.pixeltable.com/integrations/frameworks). Not Airflow, full reprocesses, or custom retry glue.

```python
# LLM provider
t.add_computed_column(
    summary=openai.chat_completions(
        messages=[{'role': 'user', 'content': t.text}],
        model='gpt-4o-mini',
    ),
)

# Local model inference
t.add_computed_column(
    classification=huggingface.vit_for_image_classification(t.image),
)

# Multimodal vision
t.add_computed_column(
    description=openai.chat_completions(
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': 'Describe this image'},
                    {'type': 'image_url', 'image_url': t.image},
                ],
            },
        ],
        model='gpt-4o-mini',
    ),
)
```

[Computed columns](https://docs.pixeltable.com/tutorials/computed-columns) · [Built-ins](https://docs.pixeltable.com/sdk/latest/pixeltable) · [AI integrations](https://docs.pixeltable.com/integrations/frameworks)
</details>

<details>
<summary><b>Extend:</b> your code, with cache and retry</summary>
<br>

`@pxt.udf` and `@pxt.query` with parallelize, cache, and retry. Not one-off handlers with no cache or retry.

```python
@pxt.udf
def format_prompt(context: list, question: str) -> str:
    return f'Context: {context}\nQuestion: {question}'


@pxt.query
def search_by_topic(topic: str):
    return t.where(t.category == topic).select(t.title, t.summary)
```

[UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Custom aggregates](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda)
</details>

<details>
<summary><b>Index:</b> built-in vector search</summary>
<br>

`add_embedding_index()` stays in sync with table data. Query with `.similarity()` on any indexed column. Not Pinecone, pgvector, and manual ETL.

```python
t.add_embedding_index(
    'img',
    embedding=clip.using(model_id='openai/clip-vit-base-patch32'),
)

sim = t.img.similarity(string='cat playing with yarn')
results = t.order_by(sim, asc=False).limit(10).collect()
```

[Embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Semantic search](https://docs.pixeltable.com/howto/cookbooks/search/search-semantic-text) · [Image search app](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi)
</details>

<details>
<summary><b>Query & experiment:</b> prototype to production in one line</summary>
<br>

`.select()` and `.sample()` to test UDFs ephemerally; same expression becomes `add_computed_column()` when ready. Not notebook experiments rewritten for production.

```python
# Explore: filter, sample, apply UDFs ephemerally
results = (
    t.where(t.score > 0.8)
    .order_by(t.timestamp)
    .select(t.image, score=t.score)
    .limit(10)
    .collect()
)

# Test on a sample (nothing stored, parallelized and cached)
t.sample(5).select(t.text, summary=summarize(t.text)).collect()

# Commit: same expression, full dataset, skips cached rows
t.add_computed_column(summary=summarize(t.text))
```

[Queries & expressions](https://docs.pixeltable.com/tutorials/queries-and-expressions) · [Iterative workflow](https://docs.pixeltable.com/howto/cookbooks/core/dev-iterative-workflow)
</details>

<details>
<summary><b>Agents & tools:</b> tool calling and MCP</summary>
<br>

`pxt.tools()`, `invoke_tools()`, and MCP: LLMs choose what to invoke and Pixeltable stores results. Not LangChain loops and manual tool wiring.

```python
mcp_tools = pxt.mcp_udfs('http://localhost:8000/mcp')
tools = pxt.tools(get_weather_udf, search_context_query, *mcp_tools)

t.add_computed_column(
    tool_output=invoke_tools(tools, t.llm_tool_choice),
)
```

[Tool calling](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling) · [Agents & MCP](https://docs.pixeltable.com/use-cases/agents-mcp)
</details>

<details>
<summary><b>Serve:</b> HTTP from schema</summary>
<br>

`pxt serve` from TOML or `FastAPIRouter` routes on the same app. Not hand-written FastAPI endpoints for every table operation.

```toml
# pyproject.toml
[[tool.pixeltable.service]]
name = "my-service"
modules = ["schema"]

[[tool.pixeltable.service.routes]]
type = "insert"
table = "myapp.docs"
path = "/ingest"
inputs = ["document"]
outputs = ["document", "summary"]
```

```bash
pxt serve my-service
```

```python
from pixeltable.serving import FastAPIRouter

router = FastAPIRouter(prefix='/api', tags=['data'])
router.add_query_route(path='/search', query=search_documents)
router.add_insert_route(table, path='/upload', uploadfile_inputs=['image'])
```

[CLI serving](https://docs.pixeltable.com/platform/cli) · [Deployment overview](https://docs.pixeltable.com/howto/deployment/overview)
</details>

<details>
<summary><b>Inspect & visualize:</b> errors, tables, and pipelines</summary>
<br>

`pxt errors` and queryable `errormsg` per cell; `pxt dashboard` opens a local UI to browse tables, preview media, and trace column lineage. Not log scraping or opaque per-row failures.

```bash
pxt errors my_table          # rows where a computed column failed
pxt dashboard                # browse tables, preview media, pipeline graph
```

Table browser · media lightbox · column lineage · per-column errors · CSV export

[CLI](https://docs.pixeltable.com/platform/cli) · [Dashboard](https://docs.pixeltable.com/platform/dashboard)
</details>

<details>
<summary><b>Version:</b> time travel</summary>
<br>

`history()`, `revert()`, and snapshot queries for time travel on every insert and schema change. Not DVC, MLflow, and backfill scripts.

```python
t = pxt.get_table('my_table')
t.revert()  # undo last modification
t.history()  # list all versions
snapshot = pxt.get_table('my_table:472')  # query a snapshot
```

[Version control](https://docs.pixeltable.com/platform/version-control)
</details>

<br>

**Three deployment patterns** ([docs](https://docs.pixeltable.com/howto/deployment/overview) / [starter kit](https://github.com/pixeltable/pixeltable-starter-kit)):

| Pattern | What it is | You write |
|---|---|---|
| **Full Backend** | FastAPI + React web app | Python schema + endpoints + frontend |
| **Batch Processing** | Cron / queue / Cloud Run Job | Python script: ingest, compute, `export_sql`, exit |
| **Declarative API** | REST API from TOML config | `pyproject.toml` routes + `pxt serve` |

---

## Installation

```bash
pip install pixeltable  # SDK + CLI (pxt ls, rows, errors, …)
```

## AI Agent Skill

Teach AI coding assistants (Cursor, Claude Code, Copilot, etc.). [Learn more →](https://github.com/pixeltable/pixeltable-skill)

```bash
npx skills add pixeltable/pixeltable-skill
```

## Start from a Template

Head start on a production-ready app: scaffold schema, routes, and deployment pattern in one command.

```bash
uvx pixeltable-new myapp
```

Default: declarative serving (`schema.py` + `pyproject.toml` → `pxt serve`). `--backend` for FastAPI + React; `--batch` for cron/queue scripts. Templates from the [Starter Kit](https://github.com/pixeltable/pixeltable-starter-kit).

## Quick Start

Define schema in Python, routes in TOML: a `pxt.Video` table, frame view, one computed column on the frame view, and a single insert endpoint.

```python
# demo.py
import pixeltable as pxt
from pixeltable.functions.video import frame_iterator

videos = pxt.create_table('videos', {'video': pxt.Video, 'title': pxt.String}, if_exists='ignore')
frames = pxt.create_view('frames', videos, iterator=frame_iterator(videos.video, fps=1), if_exists='ignore')
frames.add_computed_column(thumb=frames.frame.thumbnail((320, 320)), if_exists='ignore')
```

```toml
# pyproject.toml
[[tool.pixeltable.service]]
name = "video-api"
modules = ["demo"]

[[tool.pixeltable.service.routes]]
type = "insert"
path = "/videos"
table = "videos"
inputs = ["video", "title"]
outputs = ["title"]
```

```bash
python demo.py   # create tables, views, and computed columns
pxt serve video-api
# curl -X POST localhost:8000/videos -H 'Content-Type: application/json' -d '{"video": "https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources/bangkok.mp4", "title": "Bangkok"}'
pxt ls -l && pxt rows frames -n 3 && pxt errors frames
```

See [CLI serving](https://docs.pixeltable.com/platform/cli).

## Demo

See Pixeltable in action: table creation, computed columns, multimodal processing, and querying in a single workflow.

https://github.com/user-attachments/assets/b50fd6df-5169-4881-9dbe-1b6e5d06cede

## Documentation

One schema for storage, orchestration, and retrieval. [What is Pixeltable?](https://docs.pixeltable.com/overview/pixeltable) · [Deployment overview](https://docs.pixeltable.com/howto/deployment/overview)

| Topic | Guides |
|---|---|
| **Schema & orchestration** | [Type system](https://docs.pixeltable.com/platform/type-system) · [Tables & data](https://docs.pixeltable.com/tutorials/tables-and-data-operations) · [Computed columns](https://docs.pixeltable.com/tutorials/computed-columns) · [Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [Embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Queries & expressions](https://docs.pixeltable.com/tutorials/queries-and-expressions) · [Iterative workflow](https://docs.pixeltable.com/howto/cookbooks/core/dev-iterative-workflow) · [Version control](https://docs.pixeltable.com/platform/version-control) |
| **Agents & serving** | [Agents & MCP](https://docs.pixeltable.com/use-cases/agents-mcp) · [Tool calling](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling) · [RAG pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) · [CLI & dashboard](https://docs.pixeltable.com/platform/cli) · [UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Built-ins](https://docs.pixeltable.com/sdk/latest/pixeltable) · [30+ providers](https://docs.pixeltable.com/integrations/frameworks) |
| **Cloud & storage** | [Cloud storage](https://docs.pixeltable.com/integrations/cloud-storage) (S3, GCS, Azure, R2, B2, Tigris) · [Configuration](https://docs.pixeltable.com/platform/configuration) · [External files](https://docs.pixeltable.com/platform/external-files) · [Get started](https://docs.pixeltable.com/use-cases/get-started) · [Cloud services](https://docs.pixeltable.com/use-cases/services) · [Public datasets](https://www.pixeltable.com/data-products) |
| **Local & I/O** | [Storage architecture](https://docs.pixeltable.com/howto/deployment/infrastructure#storage-architecture) · [CSV import](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [Hugging Face](https://docs.pixeltable.com/howto/cookbooks/data/data-import-huggingface) · [PyTorch export](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch) · [ML data wrangling](https://docs.pixeltable.com/use-cases/ml-data-wrangling) · [Sample apps](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps) · [Colab tour](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/overview/ten-minute-tour.ipynb) |

## Contributing

We love contributions! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code changes, please check out our [Contributing Guide](CONTRIBUTING.md) and join our [Discord Server](https://discord.gg/QPyqFYx2UN).

## License

Pixeltable is licensed under the [Apache 2.0 License](https://opensource.org/licenses/Apache-2.0).

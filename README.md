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

## Make Building Multimodal AI Apps Dead Simple

**The only multimodal backend where AI transformations live in the schema, not bolted on top.** Your table schema is the infrastructure spec: declare tables, views, computed columns, and indexes; storage, transforms, embeddings, agents, and serving follow from it.

### What you need

- **Store** multimodal data: `pxt.create_table()` with native media types + `destination=` (not S3 + Postgres + boto3 sync)
- **Import / export**: `import_csv()`, Hugging Face, `export_parquet()`, PyTorch (not per-format ETL scripts)
- **Iterate** media into rows: `create_view()` + `FrameIterator` / `DocumentSplitter` (not FFmpeg/spaCy + child tables + FKs)
- **Orchestrate** on changes: `add_computed_column()` with incremental recompute on stale cells only (not Airflow + full reprocess + retry glue)
- **Extend** with your code: `@pxt.udf` / `@pxt.query` with parallelize, cache, retry (not handlers, no cache or retry)
- **Index** embeddings: `add_embedding_index()`, always in sync (not Pinecone / pgvector + manual ETL)
- **Query** & experiment: `.select()` / `.sample()` → `add_computed_column()` (not notebook → rewrite for production)
- **Agents** & tools: `pxt.tools()` + `invoke_tools()` / MCP (not LangChain loops + tool wiring)
- **Serve** endpoints: `pxt serve` or `FastAPIRouter` (not hand-written FastAPI routes)
- **Inspect** & debug: `pxt errors`, queryable `errormsg` per cell + [dashboard](https://docs.pixeltable.com/platform/dashboard) (not log scraping, no per-row failures)
- **Version** & rollback: `history()`, `revert()` for time travel (not DVC / MLflow + backfill scripts)

## Core Capabilities

Expand any row for a quick example and doc links.

<details>
<summary><b>Store:</b> unified multimodal interface</summary>
<br>

[`pxt.Image`](https://docs.pixeltable.com/platform/type-system), `pxt.Video`, `pxt.Audio`, `pxt.Document`, `pxt.Json`: one table for structured and media data. Set `destination=` per column or globally for S3, GCS, Azure, R2, and more.

```python
t = pxt.create_table('media', {'img': pxt.Image, 'video': pxt.Video, 'doc': pxt.Document})
```

[Type system](https://docs.pixeltable.com/platform/type-system) · [Tables & data](https://docs.pixeltable.com/tutorials/tables-and-data-operations) · [Cloud storage](https://docs.pixeltable.com/integrations/cloud-storage)
</details>

<details>
<summary><b>Import / export:</b> I/O without glue scripts</summary>
<br>

Import from CSV, URLs, Hugging Face; export to Parquet, PyTorch, COCO, and more.

```python
t.insert(pxt.io.import_csv('data.csv'))
pxt.io.export_parquet(t, 'out.parquet')
ds = t.to_pytorch_dataset('image', 'label')
```

[CSV import](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [Hugging Face](https://docs.pixeltable.com/howto/cookbooks/data/data-import-huggingface) · [PyTorch export](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch) · [ML data wrangling](https://docs.pixeltable.com/use-cases/ml-data-wrangling)
</details>

<details>
<summary><b>Iterate:</b> explode media into rows</summary>
<br>

Views with iterators split videos into frames, documents into chunks, audio into segments.

```python
from pixeltable.functions.video import frame_iterator

frames = pxt.create_view('frames', videos, iterator=frame_iterator(videos.video, fps=1))
```

[Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [RAG pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline)
</details>

<details>
<summary><b>Orchestrate:</b> declarative computed columns</summary>
<br>

Define transforms once; incremental recompute runs on new or stale rows only. Built-ins cover media processing, embeddings, and [30+ providers](https://docs.pixeltable.com/integrations/frameworks).

```python
t.add_computed_column(summary=openai.chat_completions(
    messages=[{'role': 'user', 'content': t.text}], model='gpt-4o-mini'))
```

[Computed columns](https://docs.pixeltable.com/tutorials/computed-columns) · [Built-ins](https://docs.pixeltable.com/sdk/latest/pixeltable)
</details>

<details>
<summary><b>Extend:</b> your code, with cache and retry</summary>
<br>

Wrap domain logic in `@pxt.udf`, reusable reads in `@pxt.query`, batch logic in `@pxt.uda`.

```python
@pxt.udf
def format_prompt(context: list, question: str) -> str:
    return f'Context: {context}\nQuestion: {question}'
```

[UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Custom aggregates](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda)
</details>

<details>
<summary><b>Index:</b> built-in vector search</summary>
<br>

Embedding indexes stay in sync with table data. Query with `.similarity()` on any indexed column.

```python
t.add_embedding_index('text', string_embed=embed_fn)
sim = t.text.similarity(string='query')
results = t.order_by(sim, asc=False).limit(10).collect()
```

[Embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Semantic search](https://docs.pixeltable.com/howto/cookbooks/search/search-semantic-text)
</details>

<details>
<summary><b>Query & experiment:</b> prototype to production in one line</summary>
<br>

Filter, sample, and test UDFs ephemerally; promote the same expression to a stored computed column when ready.

```python
t.sample(5).select(t.text, summary=summarize(t.text)).collect()  # experiment
t.add_computed_column(summary=summarize(t.text))                   # commit
```

[Queries & expressions](https://docs.pixeltable.com/tutorials/queries-and-expressions) · [Iterative workflow](https://docs.pixeltable.com/howto/cookbooks/core/dev-iterative-workflow)
</details>

<details>
<summary><b>Agents & tools:</b> tool calling and MCP</summary>
<br>

Register UDFs, queries, or MCP tools; LLMs choose what to invoke and Pixeltable stores results.

```python
tools = pxt.tools(search_docs, get_weather_udf, *pxt.mcp_udfs('http://localhost:8000/mcp'))
t.add_computed_column(output=invoke_tools(tools, t.llm_tool_choice))
```

[Tool calling](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling) · [Agents & MCP](https://docs.pixeltable.com/use-cases/agents-mcp)
</details>

<details>
<summary><b>Serve:</b> HTTP from schema</summary>
<br>

Declarative routes in TOML (`pxt serve`) or programmatic routes with `FastAPIRouter` on the same app.

```toml
[[tool.pixeltable.service.routes]]
type = "insert"
table = "myapp.docs"
path = "/ingest"
inputs = ["document"]
outputs = ["document", "summary"]
```

[CLI serving](https://docs.pixeltable.com/platform/cli) · [Deployment overview](https://docs.pixeltable.com/howto/deployment/overview)
</details>

<details>
<summary><b>Inspect:</b> CLI and dashboard</summary>
<br>

Per-cell errors are queryable. Browse tables, preview media, and trace column lineage locally.

```bash
pxt errors my_table          # rows where a computed column failed
pxt dashboard                # open local UI
```

[CLI](https://docs.pixeltable.com/platform/cli) · [Dashboard](https://docs.pixeltable.com/platform/dashboard)
</details>

<details>
<summary><b>Version:</b> time travel</summary>
<br>

Every insert and schema change is versioned. Inspect history, revert, or query a snapshot.

```python
t.history()
t.revert()
snapshot = pxt.get_table('my_table:472')
```

[Version control](https://docs.pixeltable.com/platform/version-control)
</details>

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

## Create a New Project

Scaffold a production-ready Pixeltable project in one command:

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

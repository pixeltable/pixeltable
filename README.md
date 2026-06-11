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
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)

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

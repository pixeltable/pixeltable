<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c5ab123e-806c-49bf-93e7-151353719b16">
  <img alt="Pixeltable Logo" src="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131" width="40%">
</picture>

<div>
<br>
</div>

The only open source Python library providing declarative, transactional data infrastructure for building multimodal AI applications — with incremental storage, transformation, indexing, retrieval, and orchestration of data, all with full operational integrity.

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![Python](https://img.shields.io/pypi/pyversions/pixeltable)](https://pypi.org/project/pixeltable/)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.gg/QPyqFYx2UN)

[**Quick Start**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**API Reference**](https://docs.pixeltable.com/sdk/latest/pixeltable) |
[**Starter Kit**](https://github.com/pixeltable/pixeltable-starter-kit) |
[**AI Coding Skill**](https://github.com/pixeltable/pixeltable-skill) |
[**Pixeltable Cloud**](https://www.pixeltable.com/)

---

## Installation

```bash
pip install pixeltable
```

Pixeltable bundles its own transactional database, orchestration engine, and local dashboard. No Docker, no external services — `pip install` is all you need. All data is managed in `~/.pixeltable` and accessed through the [Python SDK](https://docs.pixeltable.com/). See [Working with External Files](https://docs.pixeltable.com/platform/external-files) and [Storage Architecture](https://docs.pixeltable.com/howto/deployment/infrastructure#storage-architecture) for details.

## Quick Start

Define your data processing and AI workflow declaratively using
**[computed columns](https://docs.pixeltable.com/tutorials/computed-columns)** on
**[tables](https://docs.pixeltable.com/tutorials/tables-and-data-operations)**.
Focus on your logic, not the data plumbing.

```python
import pixeltable as pxt
from pixeltable.functions import gemini, huggingface

# 1. Store — structured data + media references, versioned and materialized automatically
videos = pxt.create_table('video_search', {'video': pxt.Video, 'title': pxt.String})

# 2. Orchestrate — computed columns are nodes in the table's DAG; the table is the pipeline
videos.add_computed_column(scenes=videos.video.scene_detect_adaptive())

# 3. AI integration — external API calls with rate limiting, retry, and async parallelism
videos.add_computed_column(
    response=gemini.generate_content(
        [videos.video, 'Describe this video in detail.'], model='gemini-3-flash-preview'
    )
)

# 4. JSON path expressions — extract nested fields with just-in-time typing
videos.add_computed_column(
    description=videos.response.candidates[0].content.parts[0].text
)

# 5. Incremental index maintenance — embedding indexes stay in sync, no ETL pipeline needed
videos.add_embedding_index('video', embedding=gemini.embed_content.using(model='gemini-embedding-2-preview'))

# Insert data — triggers the full pipeline automatically
base_url = 'https://raw.githubusercontent.com/pixeltable/pixeltable/release/docs/resources'
videos.insert([
    {'video': f'{base_url}/bangkok.mp4', 'title': 'Bangkok Street Tour'},
    {'video': f'{base_url}/The-Pursuit-of-Happiness-Video-Extract.mp4', 'title': 'The Pursuit of Happiness'},
])

# 6. Retrieve — structured + unstructured data side by side, with on-the-fly transforms
videos.select(
    videos.video,
    videos.title,
    videos.description,
    detections=huggingface.detr_for_object_detection(
        videos.video.extract_frame(timestamp=2.0),
        model_id='facebook/detr-resnet-50',
    ),
).collect()

# 7. Cross-modal search — find similar videos using a reference image, with filters
sim = videos.video.similarity(image=f'{base_url}/The-Pursuit-of-Happiness-Screenshot.png')
videos.where(videos.description != None).order_by(sim, asc=False).limit(5).collect()
```

## What Pixeltable Does

| You Write | Pixeltable Does |
|-----------|-----------------|
| `pxt.Image`, `pxt.Video`, `pxt.Document` columns | Stores media, handles formats, caches from URLs |
| `add_computed_column(fn(...))` | Runs incrementally, caches results, retries failures |
| `add_embedding_index(column)` | Manages vector storage, keeps index in sync |
| `@pxt.udf` / `@pxt.query` | Creates reusable functions with dependency tracking |
| `table.insert(...)` | Triggers all dependent computations automatically |
| `t.sample(5).select(t.text, summary=udf(t.text))` | Experiment on a sample — nothing stored, calls parallelized and cached |
| `table.select(...).collect()` | Returns structured + unstructured data together |
| *(nothing — it's automatic)* | Versions all data and schema changes for time-travel |

That single workflow replaces most of the typical AI stack:

| Instead of ... | Pixeltable gives you ... |
|---|---|
| PostgreSQL / MySQL | `pxt.create_table()` — schema is Python, versioned automatically |
| pgAdmin / Retool / custom dashboards | Built-in local dashboard — auto-launches, zero config |
| Pinecone / Weaviate / Qdrant | `add_embedding_index()` — one line, stays in sync |
| S3 / boto3 / blob storage | `pxt.Image` / `Video` / `Audio` / `Document` types with caching; `destination='s3://...'` for cloud routing |
| Airflow / Prefect / Celery | Computed columns trigger on insert — no orchestrator needed |
| LangChain / LlamaIndex (RAG) | `@pxt.query` + `.similarity()` + computed column chaining |
| pandas / polars (multimodal) | `.sample()`, ephemeral UDFs, then `add_computed_column()` to commit — prototype to production |
| DVC / MLflow / W&B | Built-in `history()`, `revert()`, time travel (`table:N`), snapshots |
| Custom retry / rate-limit / caching | Built into every AI integration; results cached, only new rows recomputed |
| Custom ETL / glue code | Declarative schema — Pixeltable handles execution, caching, incremental updates |

On top of these, Pixeltable ships with [built-in functions](https://docs.pixeltable.com/sdk/latest/pixeltable) for media processing (FFmpeg, Pillow, spaCy), embeddings (sentence-transformers, CLIP), and [30+ AI providers](https://docs.pixeltable.com/integrations/frameworks) (OpenAI, Anthropic, Gemini, Ollama, and more). For anything domain-specific, wrap your own logic with [`@pxt.udf`](https://docs.pixeltable.com/platform/udfs-in-pixeltable). You still write the application layer (FastAPI, React, Docker).

**Deployment options:** Pixeltable can serve as your [full backend](https://docs.pixeltable.com/howto/deployment/overview) (managing media locally or syncing with S3/GCS/Azure, plus built-in vector search and orchestration) or as an [orchestration layer](https://docs.pixeltable.com/howto/deployment/overview) alongside your existing infrastructure.

## Demo

See Pixeltable in action — table creation, computed columns, multimodal processing, and querying in a single workflow:

https://github.com/user-attachments/assets/b50fd6df-5169-4881-9dbe-1b6e5d06cede

## Core Capabilities

<details>
<summary><b>Store:</b> Unified Multimodal Interface</summary>
<br>

[`pxt.Image`](https://docs.pixeltable.com/platform/type-system), `pxt.Video`, `pxt.Audio`, `pxt.Document`, `pxt.Json` – manage diverse data consistently.

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

→ [Type System](https://docs.pixeltable.com/platform/type-system) · [Tables & Data](https://docs.pixeltable.com/tutorials/tables-and-data-operations)
</details>

<details>
<summary><b>Orchestrate:</b> Declarative Computed Columns</summary>
<br>

[Define processing steps once](https://docs.pixeltable.com/tutorials/computed-columns); they run automatically on new/updated data. Supports **API calls** (OpenAI, Anthropic, Gemini), **local inference** (Hugging Face, YOLOX, Whisper), **vision models**, and any Python logic.

```python
# LLM API call
t.add_computed_column(
    summary=openai.chat_completions(
        messages=[{'role': 'user', 'content': t.text}], model='gpt-4o-mini'
    )
)

# Local model inference
t.add_computed_column(
    classification=huggingface.vit_for_image_classification(t.image)
)

# Vision analysis (multimodal)
t.add_computed_column(
    description=openai.chat_completions(
        messages=[{'role': 'user', 'content': [
            {'type': 'text', 'text': 'Describe this image'},
            {'type': 'image_url', 'image_url': t.image},
        ]}],
        model='gpt-4o-mini'
    )
)
```

→ [Computed Columns](https://docs.pixeltable.com/tutorials/computed-columns) · [AI Integrations](https://docs.pixeltable.com/integrations/frameworks) · [Sample App: Prompt Studio](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/prompt-engineering-studio-gradio-application)
</details>

<details>
<summary><b>Iterate:</b> Explode & Process Media</summary>
<br>

[Create views with iterators](https://docs.pixeltable.com/platform/views) to explode one row into many (video→frames, doc→chunks, audio→segments).

```python
from pixeltable.functions.video import frame_iterator
from pixeltable.functions.document import document_splitter

# Document chunking with overlap & metadata
chunks = pxt.create_view(
    'chunks', docs,
    iterator=document_splitter(
        document=docs.doc,
        separators='sentence,token_limit',
        overlap=50, limit=500
    )
)

# Video frame extraction
frames = pxt.create_view(
    'frames', videos,
    iterator=frame_iterator(video=videos.video, fps=0.5)
)
```

→ [Views](https://docs.pixeltable.com/platform/views) · [Iterators](https://docs.pixeltable.com/platform/iterators) · [RAG Pipeline](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline)
</details>

<details>
<summary><b>Index:</b> Built-in Vector Search</summary>
<br>

[Add embedding indexes](https://docs.pixeltable.com/platform/embedding-indexes) and perform similarity searches directly on tables/views.

```python
t.add_embedding_index(
    'img',
    embedding=clip.using(model_id='openai/clip-vit-base-patch32')
)

sim = t.img.similarity(string='cat playing with yarn')
results = t.order_by(sim, asc=False).limit(10).collect()
```

→ [Embedding Indexes](https://docs.pixeltable.com/platform/embedding-indexes) · [Semantic Search](https://docs.pixeltable.com/howto/cookbooks/search/search-semantic-text) · [Image Search App](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi)
</details>

<details>
<summary><b>Extend:</b> Bring Your Own Code</summary>
<br>

[Extend Pixeltable](https://docs.pixeltable.com/platform/udfs-in-pixeltable) with UDFs, reusable queries, batch processing, and custom aggregators.

```python
@pxt.udf
def format_prompt(context: list, question: str) -> str:
    return f'Context: {context}\nQuestion: {question}'

@pxt.query
def search_by_topic(topic: str):
    return t.where(t.category == topic).select(t.title, t.summary)
```

→ [UDFs Guide](https://docs.pixeltable.com/platform/udfs-in-pixeltable) · [Custom Aggregates](https://docs.pixeltable.com/howto/cookbooks/core/custom-aggregates-uda)
</details>

<details>
<summary><b>Agents & Tools:</b> Tool Calling & MCP Integration</summary>
<br>

Register [`@pxt.udf`](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling), `@pxt.query` functions, or **MCP servers** as callable tools. LLMs decide which tool to invoke; Pixeltable executes and stores results.

```python
# Load tools from MCP server, UDFs, and query functions
mcp_tools = pxt.mcp_udfs('http://localhost:8000/mcp')
tools = pxt.tools(get_weather_udf, search_context_query, *mcp_tools)

# LLM decides which tool to call; Pixeltable executes it
t.add_computed_column(
    tool_output=invoke_tools(tools, t.llm_tool_choice)
)
```

→ [Tool Calling Cookbook](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling) · [Agents & MCP](https://docs.pixeltable.com/use-cases/agents-mcp) · [Pixelbot](https://github.com/pixeltable/pixelbot) · [Pixelagent](https://github.com/pixeltable/pixelagent)
</details>

<details>
<summary><b>Query & Experiment:</b> The Best Path from Prototype to Production</summary>
<br>

Unlike pandas/polars, Pixeltable [persists everything](https://docs.pixeltable.com/tutorials/queries-and-expressions), parallelizes API calls automatically, caches results, and turns your experiment into production with one line change. **No separate notebook → pipeline handoff:**

```python
# Explore with a familiar DSL — filter, sample, apply UDFs ephemerally
results = (
    t.where(t.score > 0.8)
    .order_by(t.timestamp)
    .select(t.image, score=t.score)
    .limit(10)
    .collect()
)

# Sample 5 rows and test a UDF — nothing stored, API calls parallelized and cached
t.sample(5).select(t.text, summary=summarize(t.text)).collect()

# Happy? One line to commit — runs on full dataset, skips already-cached rows
t.add_computed_column(summary=summarize(t.text))
```

→ [Queries & Expressions](https://docs.pixeltable.com/tutorials/queries-and-expressions) · [Iterative Workflow](https://docs.pixeltable.com/howto/cookbooks/core/dev-iterative-workflow) · [Version Control](https://docs.pixeltable.com/platform/version-control)
</details>

<details>
<summary><b>Version:</b> Data Persistence & Time Travel</summary>
<br>

[All data is automatically stored and versioned](https://docs.pixeltable.com/platform/version-control). Query any prior version.

```python
t = pxt.get_table('my_table')  # Get a handle to an existing table
t.revert()  # Undo the last modification

t.history()  # Display all prior versions
old_version = pxt.get_table('my_table:472')  # Query a specific version
```

→ [Version Control](https://docs.pixeltable.com/platform/version-control) · [Data Sharing](https://docs.pixeltable.com/platform/data-sharing)
</details>

<details>
<summary><b>Inspect:</b> Local Dashboard</summary>
<br>

Pixeltable ships with a built-in local dashboard that launches automatically when you start a session. Browse tables, inspect schemas, view media with lightbox navigation, visualize your full data pipeline as a DAG, and track computation errors — all from your browser.

```python
import pixeltable as pxt

# Dashboard launches automatically at http://localhost:22089
pxt.init()

# Disable if needed
pxt.init(dashboard=False)
# Or set environment variable: PIXELTABLE_DASHBOARD=0
```

**Highlights:** Table browser with sorting & filtering · Media preview (images, video, audio) · Column lineage visualization · Pipeline graph · Per-column error tracking · CSV export · Auto-refresh

No extra dependencies. No setup. It's just there.
</details>

<details>
<summary><b>Import/Export:</b> I/O & Integration</summary>
<br>

[Import from any source](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) and [export to ML formats](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch).

```python
# Import from files, URLs, S3, Hugging Face
t.insert(pxt.io.import_csv('data.csv'))
t.insert(pxt.io.import_huggingface_dataset(dataset))

# Export to analytics/ML formats
pxt.io.export_parquet(table, 'data.parquet')
pytorch_ds = table.to_pytorch_dataset('pt')  # → PyTorch DataLoader ready
coco_path = table.to_coco_dataset()          # → COCO annotations

# ML tool integrations
pxt.create_label_studio_project(table, label_config)  # Annotation
pxt.export_images_as_fo_dataset(table, table.image)   # FiftyOne
```

→ [Data Import](https://docs.pixeltable.com/howto/cookbooks/data/data-import-csv) · [PyTorch Export](https://docs.pixeltable.com/howto/cookbooks/data/data-export-pytorch) · [Label Studio](https://docs.pixeltable.com/howto/using-label-studio-with-pixeltable) · [Data Wrangling for ML](https://docs.pixeltable.com/use-cases/ml-data-wrangling)
</details>

## Tutorials & Cookbooks

| Fundamentals | Cookbooks | Providers | Sample Apps |
|:-------------|:----------|:----------|:------------|
| [![Colab](https://img.shields.io/badge/10--Minute_Tour-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/overview/ten-minute-tour.ipynb) | [![Colab](https://img.shields.io/badge/Agentic_Patterns-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/cookbooks/agents/agentic-patterns.ipynb) | [![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?logo=openai&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-openai.ipynb) | [![GitHub](https://img.shields.io/badge/Starter_Kit-181717?logo=github&logoColor=white)](https://github.com/pixeltable/pixeltable-starter-kit) |
| [![Colab](https://img.shields.io/badge/Computed_Columns-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/computed-columns.ipynb) | [![Colab](https://img.shields.io/badge/RAG_Pipeline-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/rag-demo.ipynb) | [![Anthropic](https://img.shields.io/badge/Anthropic-191919?logo=anthropic&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-anthropic.ipynb) | [![GitHub](https://img.shields.io/badge/JFK_Files_MCP-181717?logo=github&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/jfk-files-mcp-server) |
| [![Colab](https://img.shields.io/badge/Tables_&_Operations-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/tables-and-data-operations.ipynb) | [![Colab](https://img.shields.io/badge/Tool--Calling_Agents-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/cookbooks/agents/llm-tool-calling.ipynb) | [![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-gemini.ipynb) | [![GitHub](https://img.shields.io/badge/Image%2FText_Search-181717?logo=github&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi) |
| [![Colab](https://img.shields.io/badge/UDFs-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/udfs-in-pixeltable.ipynb) | [![Colab](https://img.shields.io/badge/Audio_Transcription-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/audio-transcriptions.ipynb) | [![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-ollama.ipynb) | [![GitHub](https://img.shields.io/badge/Multimodal_Chat-181717?logo=github&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/multimodal-chat) |
| [![Colab](https://img.shields.io/badge/Embedding_Indexes-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/embedding-indexes.ipynb) | [![Colab](https://img.shields.io/badge/Object_Detection-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/object-detection-in-videos.ipynb) | [![DeepSeek](https://img.shields.io/badge/DeepSeek-0A6DC2?logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-deepseek.ipynb) | [![Discord](https://img.shields.io/badge/Discord_Bot-5865F2?logo=discord&logoColor=white)](https://github.com/pixeltable/pixeltable/blob/release/docs/sample-apps/context-aware-discord-bot) |
| [**All →**](https://docs.pixeltable.com/overview/ten-minute-tour) | [**All →**](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) | [**All providers →**](https://docs.pixeltable.com/integrations/frameworks) | [**All →**](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps) |

## External Storage and Pixeltable Cloud

**Supported storage providers:**

[![S3](https://img.shields.io/badge/Amazon_S3-232F3E?logo=amazons3&logoColor=white)](https://docs.pixeltable.com/integrations/cloud-storage) [![GCS](https://img.shields.io/badge/Google_Cloud-4285F4?logo=googlecloud&logoColor=white)](https://docs.pixeltable.com/integrations/cloud-storage) [![Azure](https://img.shields.io/badge/Azure_Blob-0078D4?logo=microsoftazure&logoColor=white)](https://docs.pixeltable.com/integrations/cloud-storage) [![R2](https://img.shields.io/badge/Cloudflare_R2-F38020?logo=cloudflare&logoColor=white)](https://docs.pixeltable.com/integrations/cloud-storage) [![B2](https://img.shields.io/badge/Backblaze_B2-E21E29?logo=backblaze&logoColor=white)](https://github.com/backblaze-b2-samples/b2-pixeltable-multimodal-data) [![Tigris](https://img.shields.io/badge/Tigris-00C853?logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-tigris.ipynb)

Store computed media using the `destination` parameter on columns, or set defaults globally via `PIXELTABLE_OUTPUT_MEDIA_DEST` and `PIXELTABLE_INPUT_MEDIA_DEST`. See [Configuration](https://docs.pixeltable.com/howto/configuration).

**Data Sharing:** Publish datasets to Pixeltable Cloud for team collaboration or public sharing. Replicate public datasets instantly—no account needed for replication.

```python
import pixeltable as pxt

# Replicate a public dataset (no account required)
coco = pxt.replicate(
    remote_uri='pxt://pixeltable:fiftyone/coco_mini_2017',
    local_path='coco-copy'
)

# Publish your own dataset (requires free account)
pxt.publish(source='my-table', destination_uri='pxt://myorg/my-dataset')

# Store computed media in external cloud storage
t.add_computed_column(
    thumbnail=t.image.resize((256, 256)),
    destination='s3://my-bucket/thumbnails/'
)
```

[**Data Sharing Guide**](https://docs.pixeltable.com/platform/data-sharing) | [**Cloud Storage**](https://docs.pixeltable.com/integrations/cloud-storage) | [**Public Datasets**](https://www.pixeltable.com/data-products)

## Built with Pixeltable

| Project | Description |
|:--------|:------------|
| [**Starter Kit**](https://github.com/pixeltable/pixeltable-starter-kit) | Production-ready FastAPI + React app with deployment configs for Docker, Helm, Terraform (EKS/GKE/AKS), and AWS CDK |
| [**Pixelbot**](https://github.com/pixeltable/pixelbot) | Multimodal AI agent, an interactive data studio with on-demand ML inference, media generation, and a database explore |
| [**Pixelagent**](https://github.com/pixeltable/pixelagent) | Lightweight agent framework with built-in memory and tool orchestration |
| [**Pixelmemory**](https://github.com/pixeltable/pixelmemory) | Persistent memory layer for AI applications |
| [**Pixeltable Skill**](https://github.com/pixeltable/pixeltable-skill) | AI coding skill for Cursor, Claude Code, Copilot, Windsurf, and other AI IDEs — reduces hallucination and generates accurate Pixeltable code |
| [**MCP Server**](https://github.com/pixeltable/mcp-server-pixeltable-developer) | Model Context Protocol server for Claude, Cursor, and other AI IDEs |

## Contributing

We love contributions! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code
changes, please check out our [Contributing Guide](CONTRIBUTING.md) and join the
[Discussions](https://github.com/pixeltable/pixeltable/discussions) or our
[Discord Server](https://discord.gg/QPyqFYx2UN).

## License

Pixeltable is licensed under the Apache 2.0 License.

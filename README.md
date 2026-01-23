<picture class="github-only">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131">
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/c5ab123e-806c-49bf-93e7-151353719b16">
  <img alt="Pixeltable Logo" src="https://github.com/user-attachments/assets/e9bf82b2-cace-4bd8-9523-b65495eb8131" width="40%">
</picture>

<div>
<br>
</div>

The only open source Python library providing declarative data infrastructure for building multimodal AI applications, enabling incremental storage, transformation, indexing, retrieval, and orchestration of data.

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![nightly status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![stress-tests status](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/stress-tests.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)

[**Quick Start**](https://docs.pixeltable.com/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**API Reference**](https://docs.pixeltable.com/sdk/latest/pixeltable) |
[**Sample Apps**](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps) |
[**Discord Community**](https://discord.gg/QPyqFYx2UN)

---

## Installation

```bash
pip install pixeltable
```

Pixeltable replaces the complex multi-system architecture needed for AI applications with a single declarative table interface that natively handles multimodal data like images, videos, and documents.

## Demo

https://github.com/user-attachments/assets/b50fd6df-5169-4881-9dbe-1b6e5d06cede

## Quick Start

With Pixeltable, you define your *entire* data processing and AI workflow declaratively using
**[computed columns](https://docs.pixeltable.com/tutorials/computed-columns)** on
**[tables](https://docs.pixeltable.com/tutorials/tables-and-data-operations)**.
Focus on your application logic, not the data plumbing.

```python

# Installation
pip install -qU torch transformers openai pixeltable

# Basic setup
import pixeltable as pxt

# Table with multimodal column types (Image, Video, Audio, Document)
t = pxt.create_table('images', {'input_image': pxt.Image})

# Computed columns: define transformation logic once, runs on all data
from pixeltable.functions import huggingface

# Object detection with automatic model management
t.add_computed_column(
    detections=huggingface.detr_for_object_detection(
        t.input_image,
        model_id='facebook/detr-resnet-50'
    )
)

# Extract specific fields from detection results
t.add_computed_column(detections_text=t.detections.label_text)

# OpenAI Vision API integration with built-in rate limiting and async management
from pixeltable.functions import openai

t.add_computed_column(
    vision=openai.vision(
        prompt="Describe what's in this image.",
        image=t.input_image,
        model='gpt-4o-mini'
    )
)

# Insert data directly from an external URL
# Automatically triggers computation of all computed columns
t.insert(input_image='https://raw.github.com/pixeltable/pixeltable/release/docs/resources/images/000000000025.jpg')

# Query - All data, metadata, and computed results are persistently stored
# Structured and unstructured data are returned side-by-side
results = t.select(
    t.input_image,
    t.detections_text,
    t.vision
).collect()
```

## What Pixeltable Handles

When you run the code above, Pixeltable automatically handles data storage, transformation, AI inference, vector indexing, incremental updates, and versioning. See [Key Principles](#key-principles) for details.

| You Write | Pixeltable Does |
|-----------|-----------------|
| `pxt.Image`, `pxt.Video`, `pxt.Document` columns | Stores media, handles formats, caches from URLs |
| `add_computed_column(fn(...))` | Runs incrementally, caches results, retries failures |
| `add_embedding_index(column)` | Manages vector storage, keeps index in sync |
| `@pxt.udf` / `@pxt.query` | Creates reusable functions with dependency tracking |
| `table.insert(...)` | Triggers all dependent computations automatically |
| `table.select(...).collect()` | Returns structured + unstructured data together |
| *(nothing—it's automatic)* | Versions all data and schema changes for time-travel |

**Deployment options:** Pixeltable can serve as your [full backend](https://docs.pixeltable.com/howto/deployment/overview) (managing media locally or syncing with S3/GCS/Azure, plus built-in vector search and orchestration) or as an [orchestration layer](https://docs.pixeltable.com/howto/deployment/overview) alongside your existing infrastructure.

## Where Did My Data Go?

Pixeltable workloads generate various outputs, including both structured outputs (such as bounding boxes for detected objects) and unstructured outputs (such as generated images or video). By default, everything resides in your Pixeltable user directory at `~/.pixeltable`. Structured data is stored in a Postgres instance in `~/.pixeltable`. Generated media (images, video, audio, documents) are stored outside the Postgres database, in separate flat files in `~/.pixeltable/media`. Those media files are referenced by URL in the database, and Pixeltable provides the "glue" for a unified table interface over both structured and unstructured data.

In general, the user is not expected to interact directly with the data in `~/.pixeltable`; the data store is fully managed by Pixeltable and is intended to be accessed through the [Pixeltable Python SDK](https://docs.pixeltable.com/).

See [Working with External Files](https://docs.pixeltable.com/platform/external-files) for details on loading data from URLs, S3, and local paths.

## Key Principles

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
       'metadata': pxt.Json
   }
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
       messages=[{"role": "user", "content": t.text}], model='gpt-4o-mini'
   )
)

# Local model inference
t.add_computed_column(
   classification=huggingface.vit_for_image_classification(t.image)
)

# Vision analysis
t.add_computed_column(
   description=openai.vision(prompt="Describe this image", image=t.image)
)
```

→ [Computed Columns](https://docs.pixeltable.com/tutorials/computed-columns) · [AI Integrations](https://docs.pixeltable.com/integrations/frameworks) · [Sample App: Prompt Studio](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/prompt-engineering-studio-gradio-application)
</details>

<details>
<summary><b>Iterate:</b> Explode & Process Media</summary>
<br>

[Create views with iterators](https://docs.pixeltable.com/platform/views) to explode one row into many (video→frames, doc→chunks, audio→segments).

```python
# Document chunking with overlap & metadata
chunks = pxt.create_view('chunks', docs,
   iterator=DocumentSplitter.create(
       document=docs.doc,
       separators='sentence,token_limit',
       overlap=50, limit=500
   ))

# Video frame extraction
frames = pxt.create_view('frames', videos,
   iterator=FrameIterator.create(video=videos.video, fps=0.5))
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

sim = t.img.similarity(string="cat playing with yarn")
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
   return f"Context: {context}\nQuestion: {question}"

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
<summary><b>Query & Experiment:</b> SQL-like Python Querying</summary>
<br>

[Familiar syntax](https://docs.pixeltable.com/tutorials/queries-and-expressions) combined with powerful AI capabilities. **Test transformations before committing:**

```python
# Query data
results = (
   t.where(t.score > 0.8)
   .order_by(t.timestamp)
   .select(t.image, score=t.score)
   .limit(10)
   .collect()
)

# Test transformation on sample BEFORE adding column
t.select(t.text, summary=summarize(t.text)).head(3)  # Nothing stored
t.add_computed_column(summary=summarize(t.text))      # Now commit
```

→ [Queries & Expressions](https://docs.pixeltable.com/tutorials/queries-and-expressions) · [Iterative Development](https://docs.pixeltable.com/howto/deployment/operations)
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
| [![Colab](https://img.shields.io/badge/10--Minute_Tour-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/overview/ten-minute-tour.ipynb) | [![Colab](https://img.shields.io/badge/RAG_Pipeline-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/rag-demo.ipynb) | [![OpenAI](https://img.shields.io/badge/OpenAI-74aa9c?logo=openai&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-openai.ipynb) | [![Gradio](https://img.shields.io/badge/Prompt_Studio-FF7C00?logo=gradio&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/prompt-engineering-studio-gradio-application) |
| [![Colab](https://img.shields.io/badge/Tables_&_Operations-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/tables-and-data-operations.ipynb) | [![Colab](https://img.shields.io/badge/Tool--Calling_Agents-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/cookbooks/agents/llm-tool-calling.ipynb) | [![Anthropic](https://img.shields.io/badge/Anthropic-191919?logo=anthropic&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-anthropic.ipynb) | [![GitHub](https://img.shields.io/badge/Image%2FText_Search-181717?logo=github&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi) |
| [![Colab](https://img.shields.io/badge/Computed_Columns-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/computed-columns.ipynb) | [![Colab](https://img.shields.io/badge/Object_Detection-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/object-detection-in-videos.ipynb) | [![Gemini](https://img.shields.io/badge/Gemini-8E75B2?logo=googlegemini&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-gemini.ipynb) | [![Discord](https://img.shields.io/badge/Discord_Bot-5865F2?logo=discord&logoColor=white)](https://github.com/pixeltable/pixeltable/blob/release/docs/sample-apps/context-aware-discord-bot) |
| [![Colab](https://img.shields.io/badge/UDFs-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/udfs-in-pixeltable.ipynb) | [![Colab](https://img.shields.io/badge/Embedding_Indexes-FFDE59?logo=googlecolab&logoColor=000)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/embedding-indexes.ipynb) | [![Ollama](https://img.shields.io/badge/Ollama-000000?logo=ollama&logoColor=white)](https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-ollama.ipynb) | [![Terminal](https://img.shields.io/badge/CLI_Media_Toolkit-4D4D4D?logo=gnubash&logoColor=white)](https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/cli-media-toolkit) |
| [**All →**](https://docs.pixeltable.com/overview/ten-minute-tour) | [**All →**](https://docs.pixeltable.com/howto/cookbooks/agents/pattern-rag-pipeline) | [**All →**](https://docs.pixeltable.com/integrations/frameworks) | [**All →**](https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps) |

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
| [**Pixelbot**](https://github.com/pixeltable/pixelbot) | Multimodal Infinite Memory AI Agent — a complete E2E AI app powered by Pixeltable |
| [**Pixelagent**](https://github.com/pixeltable/pixelagent) | Lightweight agent framework with built-in memory and tool orchestration |
| [**Pixelmemory**](https://github.com/pixeltable/pixelmemory) | Persistent memory layer for AI applications |
| [**MCP Server**](https://github.com/pixeltable/mcp-server-pixeltable-developer) | Model Context Protocol server for Claude, Cursor, and other AI IDEs |

## Contributing

We love contributions! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code
changes, please check out our [Contributing Guide](CONTRIBUTING.md) and join the
[Discussions](https://github.com/pixeltable/pixeltable/discussions) or our
[Discord Server](https://discord.gg/QPyqFYx2UN).

## License

Pixeltable is licensed under the Apache 2.0 License.

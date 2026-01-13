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

Pixeltable replaces the complex data plumbing typically needed for multimodal AI—ETL pipelines, vector databases, orchestration, feature stores—with a single declarative table interface that handles images, videos, audio, and documents natively.

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

**[Unified Multimodal Interface:](https://docs.pixeltable.com/platform/type-system)** `pxt.Image`,
`pxt.Video`, `pxt.Audio`, `pxt.Document`, etc. – manage diverse data consistently.

```python
t = pxt.create_table(
   'media',
   {
       'img': pxt.Image,
       'video': pxt.Video
   }
)
```

**[Declarative Computed Columns:](https://docs.pixeltable.com/tutorials/computed-columns)** Define processing
steps once; they run automatically on new/updated data.

```python
t.add_computed_column(
   classification=huggingface.vit_for_image_classification(
       t.image
   )
)
```

**[Built-in Vector Search:](https://docs.pixeltable.com/platform/embedding-indexes)** Add embedding indexes and
perform similarity searches directly on tables/views.

```python
t.add_embedding_index(
   'img',
   embedding=clip.using(
       model_id='openai/clip-vit-base-patch32'
   )
)

sim = t.img.similarity(string="cat playing with yarn")
```

**[Incremental View Maintenance:](https://docs.pixeltable.com/platform/views)** Create virtual tables using iterators
for efficient processing without data duplication.

```python
# Document chunking with overlap & metadata and many more options to build your own iterator
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

**[Seamless AI Integration:](https://docs.pixeltable.com/integrations/frameworks)** Built-in functions for
OpenAI, Anthropic, Hugging Face, CLIP, YOLOX, and more.

```python
# LLM integration (OpenAI, Anthropic, etc.)
t.add_computed_column(
   response=openai.chat_completions(
       messages=[{"role": "user", "content": t.prompt}], model='gpt-4o-mini'
   )
)

# Computer vision (YOLOX object detection)
t.add_computed_column(
   detections=yolox(t.image, model_id='yolox_s', threshold=0.5)
)

# Embedding models (Hugging Face, CLIP)
t.add_computed_column(
   embeddings=huggingface.sentence_transformer(
       t.text, model_id='all-MiniLM-L6-v2'
   )
)
```

**[Bring Your Own Code:](https://docs.pixeltable.com/platform/udfs-in-pixeltable)** Extend Pixeltable with UDFs, batch processing, and custom aggregators.

```python
@pxt.udf
def format_prompt(context: list, question: str) -> str:
   return f"Context: {context}\nQuestion: {question}"
```

**[Agentic Workflows / Tool Calling:](https://docs.pixeltable.com/howto/cookbooks/agents/llm-tool-calling)** Register `@pxt.udf`,
`@pxt.query` functions, or **MCP tools** as tools.

```python
# Example tools: UDFs, Query functions, and MCP tools
mcp_tools = pxt.mcp_udfs('http://localhost:8000/mcp')  # Load from MCP server
tools = pxt.tools(get_weather_udf, search_context_query, *mcp_tools)

# LLM decides which tool to call; Pixeltable executes it
t.add_computed_column(
   tool_output=invoke_tools(tools, t.llm_tool_choice)
)
```

**[Data Persistence:](https://docs.pixeltable.com/tutorials/tables-and-data-operations)** All data,
metadata, and computed results are automatically stored and versioned.

```python
t = pxt.get_table('my_table')  # Get a handle to an existing table
t.select(t.account, t.balance).collect()  # Query its contents
t.revert()  # Undo the last modification to the table and restore its previous state
```

**[Time Travel:](https://docs.pixeltable.com/platform/version-control)** By default,
Pixeltable preserves the full change history of each table, and any prior version can be selected and queried.

```python
t.history()  # Display a human-readable list of all prior versions of the table
old_version = pxt.get_table('my_table:472')  # Get a handle to a specific table version
old_version.select(t.account, t.balance).collect()  # Query the older version
```

**[SQL-like Python Querying:](https://docs.pixeltable.com/tutorials/queries-and-expressions)** Familiar syntax
combined with powerful AI capabilities.

```python
results = (
   t.where(t.score > 0.8)
   .order_by(t.timestamp)
   .select(t.image, score=t.score)
   .limit(10)
   .collect()
)
```

**[I/O & Integration:](https://pixeltable.github.io/pixeltable/pixeltable/io/)** Export to multiple
formats and integrate with ML/AI tools ecosystem.

```python
# Export to analytics/ML formats
pxt.export_parquet(table, 'data.parquet', partition_size_bytes=100_000_000)
pxt.export_lancedb(table, 'vector_db')

# DataFrame conversions
results = table.select(table.image, table.labels).collect()
df = results.to_pandas()                           # → pandas DataFrame
models = results.to_pydantic(MyModel)              # → Pydantic models

# Specialized ML dataset formats
coco_path = table.to_coco_dataset()                # → COCO annotations
pytorch_ds = table.to_pytorch_dataset('pt')        # → PyTorch DataLoader ready

# ML tool integrations
pxt.create_label_studio_project(table, label_config)  # Annotation
pxt.export_images_as_fo_dataset(table, table.image)   # FiftyOne
```

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

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

```python
pip install pixeltable
```
Pixeltable replaces the complex multi-system architecture typically needed for AI applications (databases, file storage, vector DBs, APIs, orchestration) with a single declarative table interface that natively handles multimodal data like images, videos, and documents.

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

## What Happened?

* **Data Ingestion & Storage:** References [files](https://docs.pixeltable.com/platform/external-files)
    (images, videos, audio, docs) in place, handles structured data.
* **Transformation & Processing:** Applies *any* Python function ([UDFs](https://docs.pixeltable.com/platform/udfs-in-pixeltable))
    or built-in operations ([chunking, frame extraction](https://docs.pixeltable.com/platform/iterators)) automatically.
* **AI Model Integration:** Runs inference ([embeddings](https://docs.pixeltable.com/platform/embedding-indexes),
    [object detection](https://docs.pixeltable.com/howto/cookbooks/images/img-detect-objects),
    [LLMs](https://docs.pixeltable.com/integrations/frameworks#cloud-llm-providers)) as part of the data pipeline.
* **Indexing & Retrieval:** Creates and manages vector indexes for fast
    [semantic search](https://docs.pixeltable.com/platform/embedding-indexes)
    alongside traditional filtering.
* **Incremental Computation:** Only [recomputes](https://docs.pixeltable.com/overview/quick-start) what's
    necessary when data or code changes, saving time and cost.
* **Versioning & Lineage:** Automatically tracks data and schema changes for reproducibility. See below for an example
    that uses "time travel" to query an older version of a table.

Pixeltable can ingest data from local storage or directly from a URL. When external media files are referenced by URL,
as in the `insert` statement above, Pixeltable caches them locally before processing. See the
[Working with External Files](https://github.com/pixeltable/pixeltable/blob/main/docs/notebooks/feature-guides/working-with-external-files.ipynb)
notebook for more details.

## Where Did My Data Go?

Pixeltable workloads generate various outputs, including both structured outputs (such as bounding boxes for detected
objects) and/or unstructured outputs (such as generated images or video). By default, everything resides in your
Pixeltable user directory at `~/.pixeltable`. Structured data is stored in a Postgres instance in `~/.pixeltable`.
Generated media (images, video, audio, documents) are stored outside the Postgres database, in separate flat files in
`~/.pixeltable/media`. Those media files are referenced by URL in the database, and Pixeltable provides the "glue" for
a unified table interface over both structured and unstructured data.

In general, the user is not expected to interact directly with the data in `~/.pixeltable`; the data store is fully
managed by Pixeltable and is intended to be accessed through the Pixeltable Python SDK.

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

## Key Examples

*(See the [Full Quick Start](https://docs.pixeltable.com/overview/quick-start) or
[Notebook Gallery](#notebook-gallery) for more details)*

**1. Multimodal Data Store and Data Transformation (Computed Column):**

```bash
pip install pixeltable
```

```python
import pixeltable as pxt

# Create a table
t = pxt.create_table(
    'films',
    {'name': pxt.String, 'revenue': pxt.Float, 'budget': pxt.Float},
    if_exists="replace"
)

t.insert([
    {'name': 'Inside Out', 'revenue': 800.5, 'budget': 200.0},
    {'name': 'Toy Story', 'revenue': 1073.4, 'budget': 200.0}
])

# Add a computed column for profit - runs automatically!
t.add_computed_column(profit=(t.revenue - t.budget), if_exists="replace")

# Query the results
print(t.select(t.name, t.profit).collect())
# Output includes the automatically computed 'profit' column
```

**2. Object Detection with [YOLOX](https://github.com/pixeltable/pixeltable-yolox):**

```bash
pip install pixeltable pixeltable-yolox
```

```python
import PIL
import pixeltable as pxt
from yolox.models import Yolox
from yolox.data.datasets import COCO_CLASSES

t = pxt.create_table('image', {'image': pxt.Image}, if_exists='replace')

# Insert some images
prefix = 'https://upload.wikimedia.org/wikipedia/commons'
paths = [
    '/1/15/Cat_August_2010-4.jpg',
    '/e/e1/Example_of_a_Dog.jpg',
    '/thumb/b/bf/Bird_Diversity_2013.png/300px-Bird_Diversity_2013.png'
]
t.insert({'image': prefix + p} for p in paths)

@pxt.udf
def detect(image: PIL.Image.Image) -> list[str]:
    model = Yolox.from_pretrained("yolox_s")
    result = model([image])
    coco_labels = [COCO_CLASSES[label] for label in result[0]["labels"]]
    return coco_labels

t.add_computed_column(classification=detect(t.image))

print(t.select().collect())
```

**3. Image Similarity Search (CLIP Embedding Index):**

```bash
pip install pixeltable sentence-transformers
```

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip

# Create image table and add sample images
images = pxt.create_table('my_images', {'img': pxt.Image}, if_exists='replace')
images.insert([
    {'img': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/1920px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg'},
    {'img': 'https://upload.wikimedia.org/wikipedia/commons/d/d5/Retriever_in_water.jpg'}
])

# Add CLIP embedding index for similarity search
images.add_embedding_index(
    'img',
    embedding=clip.using(model_id='openai/clip-vit-base-patch32')
)

# Text-based image search
query_text = "a dog playing fetch"
sim_text = images.img.similarity(string=query_text)
results_text = images.order_by(sim_text, asc=False).limit(3).select(
    image=images.img, similarity=sim_text
).collect()
print("--- Text Query Results ---")
print(results_text)
```

**4. Multimodal/Incremental RAG Workflow (Document Chunking & LLM Call):**

```bash
pip install pixeltable openai spacy sentence-transformers
```

```bash
python -m spacy download en_core_web_sm
```

```python
import pixeltable as pxt
import pixeltable.functions as pxtf
from pixeltable.functions import openai, huggingface
from pixeltable.iterators import DocumentSplitter

# Manage your tables by directories
directory = "my_docs"
pxt.drop_dir(directory, if_not_exists="ignore", force=True)
pxt.create_dir("my_docs")

# Create a document table and add a PDF
docs = pxt.create_table(f'{directory}.docs', {'doc': pxt.Document})
docs.insert([{'doc': 'https://github.com/pixeltable/pixeltable/raw/release/docs/resources/rag-demo/Jefferson-Amazon.pdf'}])

# Create chunks view with sentence-based splitting
chunks = pxt.create_view(
    'doc_chunks',
    docs,
    iterator=DocumentSplitter.create(document=docs.doc, separators='sentence')
)

# Explicitly create the embedding function object
embed_model = huggingface.sentence_transformer.using(model_id='all-MiniLM-L6-v2')
# Add embedding index using the function object
chunks.add_embedding_index('text', string_embed=embed_model)

# Define query function for retrieval - Returns a Query expression
@pxt.query
def get_relevant_context(query_text: str, limit: int = 3):
    sim = chunks.text.similarity(string=query_text)
    # Return a list of strings (text of relevant chunks)
    return chunks.order_by(sim, asc=False).limit(limit).select(chunks.text)

# Build a simple Q&A table
qa = pxt.create_table(f'{directory}.qa_system', {'prompt': pxt.String})

# 1. Add retrieved context (now a list of strings)
qa.add_computed_column(context=get_relevant_context(qa.prompt))

# 2. Format the prompt with context
qa.add_computed_column(
    final_prompt=pxtf.string.format(
        """
        PASSAGES:
        {0}

        QUESTION:
        {1}
        """,
        qa.context,
        qa.prompt
    )
)

# 4. Generate the answer using the well-formatted prompt column
qa.add_computed_column(
    answer=openai.chat_completions(
        model='gpt-4o-mini',
        messages=[{
            'role': 'user',
            'content': qa.final_prompt
        }]
    ).choices[0].message.content
)

# Ask a question and get the answer
qa.insert([{'prompt': 'What can you tell me about Amazon?'}])
print("--- Final Answer ---")
print(qa.select(qa.answer).collect())
```

## Notebook Gallery

Explore Pixeltable's capabilities interactively:

| Topic | Notebook | Topic | Notebook |
|:----------|:-----------------|:-------------------------|:---------------------------------:|
| **Fundamentals** | | **Integrations** | |
| 10-Min Tour | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/overview/ten-minute-tour.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | OpenAI | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-openai.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Tables & Ops | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/tables-and-data-operations.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Anthropic | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-anthropic.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| UDFs | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/udfs-in-pixeltable.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Together AI | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/providers/working-with-together.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Embedding Index | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/embedding-indexes.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Label Studio | <a target="_blank" href="https://docs.pixeltable.com/examples/vision/label-studio"> <img src="https://img.shields.io/badge/📚%20Docs-013056" alt="Visit Docs"/></a> |
| External Files | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/platform/external-files.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Mistral | <a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/Pixeltable/incremental_prompt_engineering_and_model_comparison.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Github"/> |
| **Use Cases** | | **Sample Apps** | |
| RAG Demo | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/rag-demo.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> | Multimodal Agent | <a target="_blank" href="https://huggingface.co/spaces/Pixeltable/Multimodal-Powerhouse"> <img src="https://img.shields.io/badge/🤗%20Demo-FF7D04" alt="HF Space"/></a> |
| Object Detection | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/object-detection-in-videos.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Image/Text Search | <a target="_blank" href="https://github.com/pixeltable/pixeltable/tree/release/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi">  <img src="https://img.shields.io/badge/🖥️%20App-black.svg" alt="GitHub App"/> |
| Audio Transcription | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/use-cases/audio-transcriptions.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> | Discord Bot | <a target="_blank" href="https://github.com/pixeltable/pixeltable/blob/release/docs/sample-apps/context-aware-discord-bot"> <img src="https://img.shields.io/badge/%F0%9F%92%AC%20Bot-%235865F2.svg" alt="GitHub App"/></a> |

## Maintaining Production-Ready Multimodal AI Apps is Still Too Hard

Building robust AI applications, especially [multimodal](https://docs.pixeltable.com/platform/type-system) ones,
requires stitching together numerous tools:

* ETL pipelines for data loading and transformation.
* Vector databases for semantic search.
* Feature stores for ML models.
* Orchestrators for scheduling.
* Model serving infrastructure for inference.
* Separate systems for parallelization, caching, versioning, and lineage tracking.

This complex "data plumbing" slows down development, increases costs, and makes applications brittle and hard to reproduce.

## Roadmap (2025)

### Cloud Infrastructure and Deployment

We're working on a hosted Pixeltable service that will:

* Enable Multimodal Data Sharing of Pixeltable Tables and Views | [Waitlist](https://www.pixeltable.com/waitlist)
* Provide a persistent cloud instance
* Turn Pixeltable workflows (Tables, Queries, UDFs) into API endpoints/[MCP Servers](https://github.com/pixeltable/pixeltable-mcp-server)

## Contributing

We love contributions! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code
changes, please check out our [Contributing Guide](CONTRIBUTING.md) and join the
[Discussions](https://github.com/pixeltable/pixeltable/discussions) or our
[Discord Server](https://discord.gg/QPyqFYx2UN).

## License

Pixeltable is licensed under the Apache 2.0 License.

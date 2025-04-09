<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/resources/pixeltable-logo-large.png"
     alt="Pixeltable Logo" width="50%" />
<br></br>

<h2>The Declarative Data Infrastructure for Multimodal AI Apps</h2>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pixeltable?logo=python&logoColor=white&)
![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-E5DDD4)
<br>
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)
<a target="_blank" href="https://huggingface.co/Pixeltable">
  <img src="https://img.shields.io/badge/🤗-HF%20Demos-FF7D04" alt="Visit our Hugging Face space"/>
</a>

[**Installation**](#-installation) |
[**Quick Start**](https://docs.pixeltable.com/docs/overview/quick-start) |
[**Documentation**](https://docs.pixeltable.com/) |
[**API Reference**](https://pixeltable.github.io/pixeltable/) |
[**Examples**](#-key-examples) |
[**Discord Community**](https://discord.gg/QPyqFYx2UN)

</div>

---

## 😩 Maintaining Production-Ready Multimodal AI Apps is Still Too Hard

Building robust AI applications, especially [multimodal](https://docs.pixeltable.com/docs/datastore/bringing-data) ones, requires stitching together numerous tools:
*   ETL pipelines for data loading and transformation.
*   Vector databases for semantic search.
*   Feature stores for ML models.
*   Orchestrators for scheduling.
*   Model serving infrastructure for inference.
*   Separate systems for parallelization, caching, versioning, and lineage tracking.

This complex "data plumbing" slows down development, increases costs, and makes applications brittle and hard to reproduce.

## ✨ What is Pixeltable?

**[Pixeltable](https://docs.pixeltable.com/docs/overview/pixeltable) is the open-source, declarative data infrastructure layer designed specifically for AI.** It eliminates the need for complex plumbing by providing a unified, Python-native interface to manage, transform, index, and query your multimodal data, seamlessly integrating AI model inference into your workflow.

With Pixeltable, you define your *entire* data processing and AI workflow declaratively using **[computed columns](https://docs.pixeltable.com/docs/datastore/computed-columns)** on **[tables](https://docs.pixeltable.com/docs/datastore/tables-and-operations)**. Pixeltable's engine then automatically handles:

*   **Data Ingestion & Storage:** References [files](https://docs.pixeltable.com/docs/datastore/bringing-data) (images, videos, audio, docs) in place, handles structured data.
*   **Transformation & Processing:** Applies *any* Python function ([UDFs](https://docs.pixeltable.com/docs/datastore/custom-functions)) or built-in operations ([chunking, frame extraction](https://docs.pixeltable.com/docs/datastore/iterators)) automatically.
*   **AI Model Integration:** Runs inference ([embeddings](https://docs.pixeltable.com/docs/datastore/embedding-index), [object detection](https://docs.pixeltable.com/docs/examples/vision/yolox), [LLMs](https://docs.pixeltable.com/docs/integrations/frameworks#cloud-llm-providers)) as part of the data pipeline.
*   **Indexing & Retrieval:** Creates and manages vector indexes for fast [semantic search](https://docs.pixeltable.com/docs/datastore/embedding-index#phase-3%3A-query) alongside traditional filtering.
*   **Incremental Computation:** Only [recomputes](https://docs.pixeltable.com/docs/overview/quick-start) what's necessary when data or code changes, saving time and cost.
*   **Versioning & Lineage:** Automatically tracks data and schema changes for reproducibility.

**Focus on your application logic, not the infrastructure.**

## 🚀 Key Features

* **Unified Multimodal Interface:** `pxt.Image`, `pxt.Video`, `pxt.Audio`, `pxt.Document`, etc. – manage diverse data consistently.
  ```python
  t = pxt.create_table('media', {'img': pxt.Image, 'video': pxt.Video})
  ```

* **Declarative Computed Columns:** Define processing steps once; they run automatically on new/updated data.
  ```python
  t.add_computed_column(classification=huggingface.vit_for_image_classification(t.image))
  ```

* **Built-in Vector Search:** Add embedding indexes and perform similarity searches directly on tables/views.
  ```python
  t.add_embedding_index('img', embedding=clip.using(model_id='openai/clip-vit-base-patch32'))
  sim = t.img.similarity("cat playing with yarn")
  ```

* **On-the-Fly Data Views:** Create virtual tables using iterators for efficient processing without data duplication.
  ```python
  frames = pxt.create_view('frames', videos, iterator=FrameIterator.create(video=videos.video, fps=1))
  ```

* **Seamless AI Integration:** Built-in functions for OpenAI, Anthropic, Hugging Face, CLIP, YOLOX, and more.
  ```python
  t.add_computed_column(response=openai.chat_completions(messages=[{"role": "user", "content": t.prompt}]))
  ```

* **Bring Your Own Code:** Extend Pixeltable with simple Python User-Defined Functions.
  ```python
  @pxt.udf
  def format_prompt(context: list, question: str) -> str:
      return f"Context: {context}\nQuestion: {question}"
  ```

* **Agentic Workflows / Tool Calling:** Register `@pxt.udf` or `@pxt.query` functions as tools and orchestrate LLM-based tool use (incl. multimodal RAG).
  ```python
  # Example tools: a UDF and a Query function for RAG
  tools = pxt.tools(get_weather_udf, search_context_query)
  # LLM decides which tool to call; Pixeltable executes it
  t.add_computed_column(tool_output=invoke_tools(tools, t.llm_tool_choice))
  ```

* **Persistent & Versioned:** All data, metadata, and computed results are automatically stored.
  ```python
  t.revert()  # Revert to a previous version
  stored_table = pxt.get_table('my_existing_table')  # Retrieve persisted table
  ```

* **SQL-like Querying:** Familiar syntax combined with powerful AI capabilities.
  ```python
  results = t.where(t.score > 0.8).order_by(t.timestamp).select(t.image, score=t.score).limit(10).collect()
  ```

## 💾 Installation

```bash
pip install pixeltable
```

**Reminder: Pixeltable acts like a database.** It stores metadata and computed results persistently, typically in a `.pixeltable` directory in your workspace. See [configuration](https://docs.pixeltable.com/docs/overview/configuration) options for Pixeltable.

## 💡 Key Examples

*(See the [Full Quick Start](https://docs.pixeltable.com/docs/overview/quick-start) or [Notebook Gallery](#-notebook-gallery) for more details)*

**1. Basic Transformation (Computed Column):**

```python
import pixeltable as pxt

# Create a table
t = pxt.create_table('films', {'name': pxt.String, 'revenue': pxt.Float, 'budget': pxt.Float})
t.insert([
  {'name': 'Inside Out', 'revenue': 800.5, 'budget': 200.0},
  {'name': 'Toy Story', 'revenue': 1073.4, 'budget': 200.0}
])

# Add a computed column for profit - runs automatically!
t.add_computed_column(profit=(t.revenue - t.budget))

# Query the results
print(t.select(t.name, t.profit).collect())
# Output includes the automatically computed 'profit' column
```

**2. Object Detection with Hugging Face:**

```python
import pixeltable as pxt
from pixeltable.functions import huggingface

# Create a table to store data persistently
t = pxt.create_table('image', {'image': pxt.Image}, if_exists='replace')

# Insert some images
prefix = 'https://upload.wikimedia.org/wikipedia/commons'
paths = [
    '/1/15/Cat_August_2010-4.jpg',
    '/e/e1/Example_of_a_Dog.jpg',
    '/thumb/b/bf/Bird_Diversity_2013.png/300px-Bird_Diversity_2013.png'
]
t.insert({'image': prefix + p} for p in paths)

# Add a computed column for image classification
t.add_computed_column(classification=huggingface.detr_for_object_detection(
    t.image,
    model_id='facebook/detr-resnet-50'
))

# Retrieve the rows where cats have been identified
print(t.select(animal = t.image,
         classification = t.classification.label_text[0]) \
.where(t.classification.label_text[0]=='cat').head())
```

**3. Image Similarity Search (CLIP Embedding Index):**

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
sim_text = images.img.similarity(query_text)
results_text = images.order_by(sim_text, asc=False).limit(3).select(
    image=images.img, similarity=sim_text
).collect()
print("--- Text Query Results ---")
print(results_text)

# Image-based image search
query_image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/Huskiesatrest.jpg/2880px-Huskiesatrest.jpg'
sim_image = images.img.similarity(query_image_url)
results_image = images.order_by(sim_image, asc=False).limit(3).select(
    image=images.img, similarity=sim_image
).collect()
print("--- Image URL Query Results ---")
print(results_image)
```

**4. Multimodal/Incremental RAG Workflow (Document Chunking & LLM Call):**

```python
import pixeltable as pxt
from pixeltable.functions import openai, huggingface
from pixeltable.iterators import DocumentSplitter

# Create a document table and add a PDF
docs = pxt.create_table('my_docs', {'doc': pxt.Document}, if_exists='replace_force')
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

# Define query function for retrieval - Returns a DataFrame expression
@pxt.query
def get_relevant_context(query_text: str, limit: int = 3):
    sim = chunks.text.similarity(query_text)
    # Return a list of strings (text of relevant chunks)
    return chunks.order_by(sim, asc=False).limit(limit).select(chunks.text)

# Simple UDF for formatting prompt with context
@pxt.udf
def format_rag_prompt(context: list[dict], question: str) -> str:
    passages = "\n\n".join(item['text'] for item in context)
    return f"""PASSAGES:

{passages}

QUESTION:

{question}"""

# Build a simple Q&A table
qa = pxt.create_table('qa_system', {'prompt': pxt.String}, if_exists='replace')

# 1. Add retrieved context (now a list of strings)
qa.add_computed_column(context=get_relevant_context(qa.prompt))

# 2. Format the prompt with context
qa.add_computed_column(final_prompt=format_rag_prompt(qa.context, qa.prompt))

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

## 🤔 Why Pixeltable?

*   **Develop Faster:** Abstract away data plumbing. Define your pipeline declaratively and let Pixeltable handle the execution, caching, and versioning.
*   **Save Costs:** Incremental computation means you only process new or changed data. Avoid expensive re-runs of embedding models or LLMs.
*   **Ensure Reproducibility:** Automatic data and code versioning provides full lineage, making experiments reproducible.
*   **Stay Flexible:** Integrate *any* Python library or model. Pixeltable orchestrates; you choose the best tools for your task.
*   **Handle Multimodality:** Natively manage and process text, images, video, and audio within a single framework.

## 🎯 Use Cases

Pixeltable is ideal for building:

*   **Retrieval-Augmented Generation (RAG):** Index and search documents, audio transcripts, video content, and combine with LLMs.
*   **Video & Image Analysis:** Extract frames, run object detection/classification, index visual content for search, transcribe audio.
*   **AI Data Pipelines:** Feature extraction, data labeling workflows, model evaluation.
*   **Agentic Systems:** Build complex workflows involving tool calling, memory, and multimodal understanding.

## 📚 Notebook Gallery

Explore Pixeltable's capabilities interactively:

| Topic | Notebook | Topic | Notebook |
|:----------|:-----------------|:-------------------------|:---------------------------------:|
| **Fundamentals** | | **Integrations** | |
| 10-Min Tour | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/pixeltable-basics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | OpenAI | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/integrations/working-with-openai.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Tables & Ops | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/fundamentals/tables-and-data-operations.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Anthropic | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/integrations/working-with-anthropic.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| UDFs | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/feature-guides/udfs-in-pixeltable.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Together AI | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/integrations/working-with-together.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Embedding Index | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/feature-guides/embedding-and-vector-indexes.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Label Studio | <a target="_blank" href="https://docs.pixeltable.com/docs/cookbooks/vision/label-studio"> <img src="https://img.shields.io/badge/📚%20Docs-013056" alt="Visit Docs"/></a> |
| External Files | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/feature-guides/working-with-external-files.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Mistral | <a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/Pixeltable/incremental_prompt_engineering_and_model_comparison.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Github"/> |
| **Use Cases** | | **Sample Apps** | |
| RAG Demo | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/rag-demo.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> | Multimodal Agent | <a target="_blank" href="https://huggingface.co/spaces/Pixeltable/Multimodal-Powerhouse"> <img src="https://img.shields.io/badge/🤗%20Demo-FF7D04" alt="HF Space"/></a> |
| Object Detection | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/object-detection-in-videos.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Image/Text Search | <a target="_blank" href="https://github.com/pixeltable/pixeltable/tree/main/docs/sample-apps/text-and-image-similarity-search-nextjs-fastapi">  <img src="https://img.shields.io/badge/🖥️%20App-black.svg" alt="GitHub App"/> |
| Audio Transcription | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/notebooks/use-cases/audio-transcriptions.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> | Discord Bot | <a target="_blank" href="https://github.com/pixeltable/pixeltable/blob/main/docs/sample-apps/context-aware-discord-bot"> <img src="https://img.shields.io/badge/%F0%9F%92%AC%20Bot-%235865F2.svg" alt="GitHub App"/></a> |

## 🤝 Contributing

We love contributions! Whether it's reporting bugs, suggesting features, improving documentation, or submitting code changes, please check out our [Contributing Guide](CONTRIBUTING.md) and join the [Discussions](https://github.com/pixeltable/pixeltable/discussions) or our [Discord Server](https://discord.gg/QPyqFYx2UN).

## 🏢 License

Pixeltable is licensed under the Apache 2.0 License.

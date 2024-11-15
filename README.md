<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable/main/docs/source/data/pixeltable-logo-large.png"
     alt="Pixeltable" width="50%" />
<br></br>

<h2>AI Data Infrastructure — Declarative, Multimodal, and Incremental</h2>

[![License](https://img.shields.io/badge/License-Apache%202.0-0530AD.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pixeltable?logo=python&logoColor=white&)
![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-E5DDD4)
<br>
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml)
[![tests status](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions/workflows/nightly.yml)
[![My Discord (1306431018890166272)](https://img.shields.io/badge/💬-Discord-%235865F2.svg)](https://discord.gg/QPyqFYx2UN)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=4D148C)](https://pypi.org/project/pixeltable/)
<a target="_blank" href="https://huggingface.co/Pixeltable">
  <img src="https://img.shields.io/badge/🤗-HF Space-FF7D04" alt="Visit our Hugging Face space"/>
</a>

[Installation](https://docs.pixeltable.com/docs/installation) |
[Documentation](https://pixeltable.readme.io/) |
[API Reference](https://pixeltable.github.io/pixeltable/) |
[Code Samples](https://github.com/pixeltable/pixeltable?tab=readme-ov-file#-code-samples) |
[Computer Vision](https://docs.pixeltable.com/docs/object-detection-in-videos) |
[LLM](https://docs.pixeltable.com/docs/document-indexing-and-rag)
</div>

Pixeltable is a Python library providing a declarative interface for multimodal data (text, images, audio, video).
It features built-in versioning, lineage tracking, and incremental updates, enabling users to **store**, **transform**,
**index**, and **iterate** on data for their ML workflows.

Data transformations, model inference, and custom logic are embedded as **computed columns**.

- **Load/Query all data types**: Interact with
    [video data](https://github.com/pixeltable/pixeltable?tab=readme-ov-file#import-media-data-into-pixeltable-videos-images-audio)
    at the [frame level](https://github.com/pixeltable/pixeltable?tab=readme-ov-file#text-and-image-similarity-search-on-video-frames-with-embedding-indexes)
    and documents at the [chunk level](https://github.com/pixeltable/pixeltable?tab=readme-ov-file#automate-data-operations-with-views-eg-split-documents-into-chunks)
- **Incremental updates for data transformation**: Maintain an
    [embedding index](https://docs.pixeltable.com/docs/embedding-vector-indexes) colocated with your data
- **Lazy evaluation and cache management**: Eliminates the need for
    [manual frame extraction](https://docs.pixeltable.com/docs/object-detection-in-videos)
- **Integrates with any Python libraries**: Use
    [built-in and custom functions (UDFs)](https://docs.pixeltable.com/docs/user-defined-functions-udfs)
    without complex pipelines
- **Data format agnostic and extensibility**: Access tables as Parquet files,
    [PyTorch datasets](https://pixeltable.github.io/pixeltable/api/data-frame/#pixeltable.DataFrame.to_pytorch_dataset),
    or [COCO annotations](https://pixeltable.github.io/pixeltable/api/table/#pixeltable.Table.to_coco_dataset)

## 💾 Installation

```python
pip install pixeltable
```

**Pixeltable is persistent. Unlike in-memory Python libraries such as Pandas, Pixeltable is a database.**

## 💡 Getting Started

Learn how to create tables, populate them with data, and enhance them with built-in or user-defined transformations.

| Topic | Notebook | Topic | Notebook |
|:----------|:-----------------|:-------------------------|:---------------------------------:|
| 10-Minute Tour of Pixeltable    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/pixeltable-basics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Tables and Data Operations    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/fundamentals/tables-and-data-operations.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| User-Defined Functions (UDFs)    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/udfs-in-pixeltable.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> | Object Detection Models | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/object-detection-in-videos.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Incremental Prompt Engineering | <a target="_blank" href="https://colab.research.google.com/github/mistralai/cookbook/blob/main/third_party/Pixeltable/incremental_prompt_engineering_and_model_comparison.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Github"/> | Working with External Files    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/howto/working-with-external-files.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |
| Integrating with Label Studio    | <a target="_blank" href="https://pixeltable.readme.io/docs/label-studio"> <img src="https://img.shields.io/badge/Documentation-013056" alt="Visit our documentation"/></a> | Audio/Video Transcript Indexing    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/audio-transcriptions.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> |
| Multimodal Application    | <a target="_blank" href="https://huggingface.co/spaces/Pixeltable/Multimodal-Powerhouse"> <img src="https://img.shields.io/badge/Hugging Face-FF7D04" alt="Visit our documentation"/></a> | Document Indexing and RAG    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/release/docs/release/tutorials/rag-demo.ipynb">  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> |

## 🧱 Code Samples

### Import media data into Pixeltable (videos, images, audio...)

```python
import pixeltable as pxt

v = pxt.create_table('external_data.videos', {'video': pxt.Video})

prefix = 's3://multimedia-commons/'
paths = [
    'data/videos/mp4/ffe/ffb/ffeffbef41bbc269810b2a1a888de.mp4',
    'data/videos/mp4/ffe/feb/ffefebb41485539f964760e6115fbc44.mp4',
    'data/videos/mp4/ffe/f73/ffef7384d698b5f70d411c696247169.mp4'
]
v.insert({'video': prefix + p} for p in paths)
```

Learn how to [work with data in Pixeltable](https://pixeltable.readme.io/docs/working-with-external-files).

### Object detection in images using DETR model

```python
import pixeltable as pxt
from pixeltable.functions import huggingface

# Create a table to store data persistently
t = pxt.create_table('image', {'image': pxt.Image})

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
t.select(animal = t.image,
         classification = t.classification.label_text[0]) \
.where(t.classification.label_text[0]=='cat').head()
```

Learn about computed columns and object detection:
[Comparing object detection models](https://pixeltable.readme.io/docs/object-detection-in-videos).

### Extend Pixeltable's capabilities with user-defined functions

```python
@pxt.udf
def draw_boxes(img: PIL.Image.Image, boxes: list[list[float]]) -> PIL.Image.Image:
    result = img.copy()  # Create a copy of `img`
    d = PIL.ImageDraw.Draw(result)
    for box in boxes:
        d.rectangle(box, width=3)  # Draw bounding box rectangles on the copied image
    return result
```

Learn more about user-defined functions:
[UDFs in Pixeltable](https://pixeltable.readme.io/docs/user-defined-functions-udfs).

### Automate data operations with views, e.g., split documents into chunks

```python
# In this example, the view is defined by iteration over the chunks of a DocumentSplitter
chunks_table = pxt.create_view(
    'rag_demo.chunks',
    documents_table,
    iterator=DocumentSplitter.create(
        document=documents_table.document,
        separators='token_limit', limit=300)
)
```

Learn how to leverage views to build your
[RAG workflow](https://pixeltable.readme.io/docs/document-indexing-and-rag).

### Evaluate model performance

```python
# The computation of the mAP metric can become a query over the evaluation output
frames_view.select(mean_ap(frames_view.eval_yolox_tiny), mean_ap(frames_view.eval_yolox_m)).show()
```

Learn how to leverage Pixeltable for [Model analytics](https://pixeltable.readme.io/docs/object-detection-in-videos).

### Working with inference services

```python
chat_table = pxt.create_table('together_demo.chat', {'input': pxt.String})

# The chat-completions API expects JSON-formatted input:
messages = [{'role': 'user', 'content': chat_table.input}]

# This example shows how additional parameters from the Together API can be used in Pixeltable
chat_table.add_computed_column(
    output=chat_completions(
        messages=messages,
        model='mistralai/Mixtral-8x7B-Instruct-v0.1',
        max_tokens=300,
        stop=['\n'],
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        repetition_penalty=1.1,
        logprobs=1,
        echo=True
    )
)
chat_table.add_computed_column(
    response=chat_table.output.choices[0].message.content
)

# Start a conversation
chat_table.insert([
    {'input': 'How many species of felids have been classified?'},
    {'input': 'Can you make me a coffee?'}
])
chat_table.select(chat_table.input, chat_table.response).head()
```

Learn how to interact with inference services such as [Together AI](https://pixeltable.readme.io/docs/together-ai) in Pixeltable.

### Text and image similarity search on video frames with embedding indexes

```python
import pixeltable as pxt
from pixeltable.functions.huggingface import clip_image, clip_text
from pixeltable.iterators import FrameIterator
import PIL.Image

video_table = pxt.create_table('videos', {'video': pxt.Video})

video_table.insert([{'video': '/video.mp4'}])

frames_view = pxt.create_view(
    'frames', video_table, iterator=FrameIterator.create(video=video_table.video))

@pxt.expr_udf
def embed_image(img: PIL.Image.Image):
    return clip_image(img, model_id='openai/clip-vit-base-patch32')

@pxt.expr_udf
def str_embed(s: str):
    return clip_text(s, model_id='openai/clip-vit-base-patch32')

# Create an index on the 'frame' column that allows text and image search
frames_view.add_embedding_index('frame', string_embed=str_embed, image_embed=embed_image)

# Now we will retrieve images based on a sample image
sample_image = '/image.jpeg'
sim = frames_view.frame.similarity(sample_image)
frames_view.order_by(sim, asc=False).limit(5).select(frames_view.frame, sim=sim).collect()

# Now we will retrieve images based on a string
sample_text = 'red truck'
sim = frames_view.frame.similarity(sample_text)
frames_view.order_by(sim, asc=False).limit(5).select(frames_view.frame, sim=sim).collect()
```

Learn how to work with [Embedding and Vector Indexes](https://docs.pixeltable.com/docs/embedding-vector-indexes).

## 🔄 AI Stack Comparison

### 🎯 Computer Vision Workflows

| Requirement | Traditional | Pixeltable |
|-------------|---------------------|------------|
| Frame Extraction | ffmpeg + custom code | Automatic via FrameIterator |
| Object Detection | Multiple scripts + caching | Single computed column |
| Video Indexing | Custom pipelines + Vector DB | Native similarity search |
| Annotation Management | Separate tools + custom code | Label Studio integration |
| Model Evaluation | Custom metrics pipeline | Built-in mAP computation |

### 🤖 LLM Workflows

| Requirement | Traditional | Pixeltable |
|-------------|---------------------|------------|
| Document Chunking | Tool + custom code | Native DocumentSplitter |
| Embedding Generation | Separate pipeline + caching | Computed columns |
| Vector Search | External vector DB | Built-in vector indexing |
| Prompt Management | Custom tracking solution | Version-controlled columns |
| Chain Management | Tool + custom code | Computed column DAGs |

### 🎨 Multimodal Workflows

| Requirement | Traditional | Pixeltable |
|-------------|---------------------|------------|
| Data Types | Multiple storage systems | Unified table interface |
| Cross-Modal Search | Complex integration | Native similarity support |
| Pipeline Orchestration | Multiple tools (Airflow, etc.) | Single declarative interface |
| Asset Management | Custom tracking system | Automatic lineage |
| Quality Control | Multiple validation tools | Computed validation columns |

## ❓ FAQ

### What is Pixeltable?

Pixeltable unifies data storage, versioning, and indexing with orchestration and model versioning under a declarative
table interface, with transformations, model inference, and custom logic represented as computed columns.

### What problems does Pixeltable solve?

Today's solutions for AI app development require extensive custom coding and infrastructure plumbing.
Tracking lineage and versions between and across data transformations, models, and deployments is cumbersome.
Pixeltable lets ML Engineers and Data Scientists focus on exploration, modeling, and app development without
dealing with the customary data plumbing.

### What does Pixeltable provide me with? Pixeltable provides:

- Data storage and versioning
- Combined Data and Model Lineage
- Indexing (e.g. embedding vectors) and Data Retrieval
- Orchestration of multimodal workloads
- Incremental updates
- Code is automatically production-ready

### Why should you use Pixeltable?

- **It gives you transparency and reproducibility**
  - All generated data is automatically recorded and versioned
  - You will never need to re-run a workload because you lost track of the input data
- **It saves you money**
  - All data changes are automatically incremental
  - You never need to re-run pipelines from scratch because you’re adding data
- **It integrates with any existing Python code or libraries**
  - Bring your ever-changing code and workloads
  - You choose the models, tools, and AI practices (e.g., your embedding model for a vector index);
    Pixeltable orchestrates the data

### What is Pixeltable not providing?

- Pixeltable is not a low-code, prescriptive AI solution. We empower you to use the best frameworks and techniques for
  your specific needs.
- We do not aim to replace your existing AI toolkit, but rather enhance it by streamlining the underlying data
  infrastructure and orchestration.

> [!TIP]
> Check out the [Integrations](https://pixeltable.readme.io/docs/working-with-openai) section, and feel free to submit
> a request for additional ones.

## 🤝 Contributing to Pixeltable

We're excited to welcome contributions from the community! Here's how you can get involved:

### 🐛 Report Issues

- Found a bug? [Open an issue](https://github.com/pixeltable/pixeltable/issues)
- Include steps to reproduce and environment details

### 💡 Submit Changes

- Fork the repository
- Create a feature branch
- Submit a [pull request](https://github.com/pixeltable/pixeltable/pulls)
- See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions

### 💬 Join the Discussion

- Have questions? Start a [Discussion](https://github.com/pixeltable/pixeltable/discussions)
- Share your Pixeltable projects and use cases
- Help others in the community

### 📝 Improve Documentation

- Suggest examples and tutorials
- Propose improvements

## 🏢 License

This library is licensed under the Apache 2.0 License.

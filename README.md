<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable/master/docs/release/pixeltable-banner.png" alt="Pixeltable" width="45%" />

# Unifying Data, Models, and Orchestration for AI Products

[![License](https://img.shields.io/badge/License-Apache%202.0-darkblue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pixeltable?logo=python&logoColor=white)
[![Platform Support](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-8A2BE2)]()
[![pytest status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions)
[![PyPI Package](https://img.shields.io/pypi/v/pixeltable?color=darkorange)](https://pypi.org/project/pixeltable/)

[Installation](https://pixeltable.github.io/pixeltable/getting-started/) | [Documentation](https://pixeltable.readme.io/) | [API Reference](https://pixeltable.github.io/pixeltable/) | [Code Samples](https://pixeltable.readme.io/recipes) | [Examples](https://github.com/pixeltable/pixeltable/tree/master/docs/release/tutorials)
</div>

Pixeltable is a Python library that lets ML Engineers and Data Scientists focus on exploration, modeling, and app development without dealing with the customary data plumbing.

### What problems does Pixeltable solve?

Today’s solutions for AI app development require extensive custom coding and infrastructure plumbing. Tracking lineage and versions between and across data transformations, models, and deployment is cumbersome.

## 💾 Installation

```python
pip install pixeltable
```
> [!NOTE]
> Check out the [Pixeltable Basics](https://pixeltable.readme.io/docs/pixeltable-basics) tutorial for a tour of its most important features.

## 💡 Getting Started
Learn how to create tables, populate them with data, and enhance them with built-in or user-defined transformations and AI operations.

| Topic | Notebook |
|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 10-Minute Tour of Pixeltable    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/tutorials/pixeltable-basics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
| User-Defined Functions (UDFs)    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/howto/udfs-in-pixeltable.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
| Comparing Object Detection Models | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/tutorials/object-detection-in-videos.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
| Experimenting with Chunking (RAG) | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/tutorials/rag-operations.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
| Working with External Files    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/howto/working-with-external-files.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>
| Integrating with Label Studio for Annotations    | <a target="_blank" href="https://pixeltable.readme.io/docs/label-studio"> <img src="https://img.shields.io/badge/Docs-Label Studio-blue" alt="Visit our documentation"/></a>

## 🧱 Code Samples

### Import media data into Pixeltable (videos, images, audio...)
```python
import pixeltable as pxt

v = pxt.create_table('external_data.videos', {'video': pxt.VideoType()})

prefix = 's3://multimedia-commons/'
paths = [
    'data/videos/mp4/ffe/ffb/ffeffbef41bbc269810b2a1a888de.mp4',
    'data/videos/mp4/ffe/feb/ffefebb41485539f964760e6115fbc44.mp4',
    'data/videos/mp4/ffe/f73/ffef7384d698b5f70d411c696247169.mp4'
]
v.insert({'video': prefix + p} for p in paths)
```
Learn how to [work with data in Pixeltable](https://pixeltable.readme.io/docs/working-with-external-files).

### Add an object detection model to your workflow
```python
table['detections'] = huggingface.detr_for_object_detection(table.input_image, model_id='facebook/detr-resnet-50')
```
Learn about computed columns and object detection: [Comparing object detection models](https://pixeltable.readme.io/docs/object-detection-in-videos).

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
Learn more about user-defined functions: [UDFs in Pixeltable](https://pixeltable.readme.io/docs/user-defined-functions-udfs).

### Automate data operations with views
```python
# In this example, the view is defined by iteration over the chunks of a DocumentSplitter.
chunks_table = pxt.create_view(
    'rag_demo.chunks',
    documents_table,
    iterator=DocumentSplitter.create(
        document=documents_table.document,
        separators='token_limit', limit=300)
)
```
Learn how to leverage views to build your [RAG workflow](https://pixeltable.readme.io/docs/document-indexing-and-rag).

### Evaluate model performance
```python
# The computation of the mAP metric can simply become a query over the evaluation output, aggregated with the mean_ap() function.
frames_view.select(mean_ap(frames_view.eval_yolox_tiny), mean_ap(frames_view.eval_yolox_m)).show()
```
Learn how to leverage Pixeltable for [Model analytics](https://pixeltable.readme.io/docs/object-detection-in-videos).

### Working with inference services
```python
chat_table = pxt.create_table('together_demo.chat', {'input': pxt.StringType()})

# The chat-completions API expects JSON-formatted input:
messages = [{'role': 'user', 'content': chat_table.input}]

# This example shows how additional parameters from the Together API can be used in Pixeltable to customize the model behavior.
chat_table['output'] = chat_completions(
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
chat_table['response'] = chat_table.output.choices[0].message.content

# Start a conversation
chat_table.insert([
    {'input': 'How many species of felids have been classified?'},
    {'input': 'Can you make me a coffee?'}
])
chat_table.select(chat_table.input, chat_table.response).head()
```
Learn how to interact with inference services such as [Together AI](https://pixeltable.readme.io/docs/together-ai) in Pixeltable.

## ❓ FAQ

### What is Pixeltable?

Pixeltable unifies data storage, versioning, and indexing with orchestration and model versioning under a declarative table interface, with transformations, model inference, and custom logic represented as computed columns.

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
  - You choose the models, tools, and AI practices (e.g., your embedding model for a vector index); Pixeltable orchestrates the data
 
### What is Pixeltable not providing?

- Pixeltable is not a low-code, prescriptive AI solution. We empower you to use the best frameworks and techniques for your specific needs.
- We do not aim to replace your existing AI toolkit, but rather enhance it by streamlining the underlying data infrastructure and orchestration.
- Pixeltable is persistent. Unlike in-memory Python libraries such as Pandas, Pixeltable is a database. When working locally or against an hosted version of Pixeltable, use [get_table](https://pixeltable.github.io/pixeltable/api/pixeltable/#pixeltable.get_table) at any time to retrieve an existing table.

> [!TIP]
> Check out the [Integrations](https://pixeltable.readme.io/docs/working-with-openai) section, and feel free to submit a request for additional ones.

## 🐛 Contributions & Feedback

Are you experiencing issues or bugs with Pixeltable? File an [Issue](https://github.com/pixeltable/pixeltable/issues).
</br>Do you want to contribute? Feel free to open a [PR](https://github.com/pixeltable/pixeltable/pulls).

## :classical_building: License

This library is licensed under the Apache 2.0 License.

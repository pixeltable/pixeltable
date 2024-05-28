<div align="center">
<img src="https://raw.githubusercontent.com/pixeltable/pixeltable/master/docs/release/pixeltable-banner.png" alt="Pixeltable" width="45%" />

# Unifying Data, Models, and Orchestration for AI Products

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
&nbsp;&nbsp;
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pixeltable?logo=python&logoColor=white)
&nbsp;&nbsp;
[![pytest status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)](https://github.com/pixeltable/pixeltable/actions)

[Installation](https://pixeltable.github.io/pixeltable/getting-started/) | [Documentation](https://pixeltable.readme.io/) | [API Reference](https://pixeltable.github.io/pixeltable/) | [Code Samples](https://pixeltable.readme.io/recipes) | [Examples](https://github.com/pixeltable/pixeltable/tree/master/docs/release/tutorials)
</div>

Pixeltable is a Python library that lets AI engineers and data scientists focus on exploration, modeling, and app development without dealing with the customary data plumbing.

## What problems does Pixeltable solve?

Today’s solutions for AI app development require extensive custom coding and infrastructure plumbing. Tracking lineage and versions between and across data transformations, models, and deployment is cumbersome. With Pixeltable you can store, transform, index, and iterate on your data within the same table interface, whether it's text, images, embeddings, or even video. Built-in lineage and versioning ensure transparency and reproducibility, while the development-to-production mirror streamlines deployment.

## 💾 Installation
Pixeltable works with Python 3.9, 3.10, 3.11, or 3.12 running on Linux, MacOS, or Windows.

```python
%pip install pixeltable
```

To verify that it's working:

```python
import pixeltable as pxt
pxt.init()
```
> [!NOTE]
> Check out the [Pixeltable Basics](https://pixeltable.readme.io/docs/pixeltable-basics) tutorial for a tour of its most important features.

## 💡 Get Started
Learn how to create tables, populate them with data, and enhance them with built-in or user-defined transformations and AI operations.

| Topic      |                                                                                Notebook                                                                                |                                                                                API                                                                                |
|:--------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Get Started    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/tutorials/pixeltable-basics.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                 [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://pixeltable.github.io/pixeltable/api/pixeltable/)                 |
| User-Defined Functions (UDFs)    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/howto/udfs-in-pixeltable.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                 [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://pixeltable.github.io/pixeltable/api/iterators/document-splitter/)                 |
| Comparing Object Detection Models | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/tutorials/object-detection-in-videos.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a>            | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://pixeltable.github.io/pixeltable/api-cheat-sheet/#frame-extraction-for-video-data) |
| Experimenting with Chunking (RAG) | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/tutorials/rag-operations.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>            | [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://pixeltable.github.io/pixeltable/api/iterators/document-splitter/) |
| Working with External Files    | <a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/release/howto/working-with-external-files.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/> </a> |                 [![API](https://img.shields.io/badge/api-reference-blue.svg)](https://pixeltable.github.io/pixeltable/api-cheat-sheet/#inserting-data-into-a-table)                 |

## ❓ FAQ

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

> [!TIP]
> Check out the [Integrations](https://pixeltable.readme.io/docs/working-with-openai) section, and feel free to submit a request for additional ones.

## 📙 Example of Use Cases

- **Interact with video data at the frame level** without having to think about frame extraction, intermediate file storage, or storage space explosion.
- **Augment your data incrementally and interactively with built-in functions and UDFs**, such as image transformations, model inference, and visualizations, without having to think about data pipelines, incremental updates, or capturing function output.
- **Interact with all the data relevant to your AI application** (video, images, documents, audio, structured data, JSON) through a simple dataframe-style API directly in Python. This includes:
  - similarity search on embeddings, supported by high-dimensional vector indexing;
  - path expressions and transformations on JSON data;
  - PIL and OpenCV image operations;
  - assembling frames into videos.
- **Perform keyword and image similarity search at the video frame level** without having to worry about frame storage.
- **Access all Pixeltable-resident data directly as a PyTorch dataset** in your training scripts.
- **Understand the compute and storage costs of your data at the granularity** of individual augmentations and get cost projections before adding new data and new augmentations.
- **Rely on Pixeltable's automatic versioning and snapshot functionality** to protect against regressions and to ensure reproducibility.

## 🐛 Contributions & Feedback

Are you experiencing issues or bugs with Pixeltable? File an [Issue](https://github.com/pixeltable/pixeltable/issues).
</br>Do you want to contribute? Feel free to open a [PR](https://github.com/pixeltable/pixeltable/pulls).

## :classical_building: License

This library is licensed under the Apache 2.0 License.

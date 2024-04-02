# Pixeltable: The AI Data Plane

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
&nbsp;&nbsp;
![pytest status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg?event=push)

Pixeltable is a Python library that lets AI engineers and data scientists focus on
exploration, modeling, and app development without having to deal with the customary
data plumbing.

**Pixeltable redefines data infrastructure and workflow orchestration for AI development.**
- **Declarative Interface:** Express logic within data using familiar operations. Pixeltable automatically orchestrates execution.
- **Multimodal Data Handling:** Perform image transformations, embed text, search video by content, and easily combine modalities.
- **Lineage Tracking:** Automatically track transformations for full reproducibility and effortless experimentation.
- **Granular Cost Accounting:** Understand and control inference costs at the level of individual transformations.
- **Model Integration & Deployment:** Load pre-trained models, perform inline inference, and prepare data for production.

## Quick Start

If you just want to play around with Pixeltable to see what it's capable of, the easiest way is to run
the Pixeltable Basics tutorial in colab:

<a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/docs/tutorials/pixeltable-basics.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Local Installation

Pixeltable works with Python 3.9, 3.10, or 3.11 running on Linux or MacOS.

```
pip install pixeltable
```

To verify that it's working:

```
import pixeltable as pxt
cl = pxt.Client()
```

For more detailed installation instructions, see the
[Getting Started with Pixeltable](https://pixeltable.github.io/pixeltable/getting-started/)
guide. Then, check out the
[Pixeltable Basics](https://pixeltable.github.io/pixeltable/tutorials/pixeltable-basics/)
tutorial for a tour of its most important features.

## Benefits of Using Pixeltable

* Interact with video data at the frame level without having to think about frame extraction,
intermediate file storage, or storage space explosion.
* Augment your data incrementally and interactively with built-in and user-defined functions such as
image transformations, model inference, visualizations, etc., without having to think about data pipelines,
incremental updates, capturing function output, etc.
* Interact with all the data relevant to your AI application (video, images, documents, audio, structured data, JSON) through
a simple dataframe-style API directly in Python. This includes:
    * similarity search on embeddings, supported by high-dimensional vector indexing
    * path expressions and transformations on JSON data
    * PIL and OpenCV image operations
    * assembling frames into videos
* Perform keyword and image similarity search at the video frame level without having to worry about frame
storage.
* Access all Pixeltable-resident data directly as a PyTorch dataset in your training scripts.
* Understand the compute and storage costs of your data at the granularity of individual augmentations and
get cost projections **before** adding new data and new augmentations.
* Rely on Pixeltable's automatic versioning and snapshot functionality to protect against regressions
and to ensure reproducibility.

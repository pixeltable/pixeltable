<img src="docs/pixeltable-banner.png" width="45%"/>

# Pixeltable: The Multimodal AI Data Plane

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
&nbsp;&nbsp;
![pytest status](https://github.com/pixeltable/pixeltable/actions/workflows/pytest.yml/badge.svg)

Pixeltable is a Python library that lets AI engineers and data scientists focus on
exploration, modeling, and app development without having to deal with the customary
data plumbing.

**Pixeltable redefines data infrastructure and workflow orchestration for AI development.**
It brings together data storage, versioning, and indexing with orchestration and model
versioning under a declarative table interface, with transformations, model inference,
and custom logic represented as computed columns.

## Quick Start

If you just want to play around with Pixeltable to see what it's capable of, the easiest way is to run
the Pixeltable Basics tutorial in colab:

<a target="_blank" href="https://colab.research.google.com/github/pixeltable/pixeltable/blob/master/docs/tutorials/pixeltable-basics.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Installation

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

## What problems does Pixeltable solve?

Today’s solutions for AI app development require extensive custom coding and infrastructure
plumbing. Tracking lineage and versions between and across data transformations, models, and
deployment is cumbersome. Pixeltable is a replacement for traditional data plumbing, providing
a unified plane for data, models, and orchestration. It removes the data plumbing overhead in
building and productionizing AI applications.

## Why should you use Pixeltable?

- It gives you transparency and reproducibility
    - All generated data is automatically recorded and versioned
    - You will never need to re-run a workload because you lost track of the input data
- It saves you money
    - All data changes are automatically incremental
    - You never need to re-run pipelines from scratch because you’re adding data
- It integrates with any existing Python code or libraries
    - Bring your ever-changing code and workloads
    - You choose the models, tools, and AI practices (e.g., your embedding model for a vector index); Pixeltable orchestrates the data

## Example Use Cases

* Interact with video data at the frame level without having to think about frame extraction,
intermediate file storage, or storage space explosion.
* Augment your data incrementally and interactively with built-in functions and UDFs, such as
image transformations, model inference, and visualizations, without having to think about data pipelines,
incremental updates, or capturing function output.
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
get cost projections before adding new data and new augmentations.
* Rely on Pixeltable's automatic versioning and snapshot functionality to protect against regressions
and to ensure reproducibility.

# Pixeltable: A Table Interface for ML Data

Pixeltable is a Python library that lets ML engineers and data scientists focus on experiments and models
without having to deal with the customary data plumbing.

* Interact with video data at the frame level without having to think about frame extraction,
intermediate file storage, or storage space explosion.
* Augment your data incrementally and interactively with built-in and user-defined functions such as
image transformations, model inference, visualizations, etc., without having to think about data pipelines,
incremental updates, capturing function output, etc.
* Interact with all the data relevant to your ML project (video, images, documents, audio, structured data, JSON) through
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

## Installation

1. Install Docker

    On MacOS: follow [these](https://docs.docker.com/desktop/install/mac-install/) instructions.

2. `pip install git+https://github.com/pixeltable/pixeltable`


## First Steps

[This tutorial](https://pixeltable.github.io/pixeltable/tutorials/pixeltable_basics/)
gives you a 10-minute overview of Pixeltable.

If you are interested in working with video and how to interact with videos at the frame level through Pixeltable,
take a look at [this tutorial](https://pixeltable.github.io/pixeltable/tutorials/comparing_object_detection_models_for_video/).


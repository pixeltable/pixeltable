# Pixeltable: A Table Interface for AI Data

Pixeltable is a Python library that lets AI engineers and data scientists focus on
exploration, modeling, and app development without having to deal with the customary
data plumbing.

## Getting Started

Visit the "Getting Started" guide for installation instructions:

* [Getting Started with Pixeltable](https://pixeltable.github.io/pixeltable/getting-started/)

Once you've installed Pixeltable, visit the "Pixeltable Basics" tutorial for a tour of its
most important features:

* [Pixeltable Basics](https://pixeltable.github.io/pixeltable/tutorials/pixeltable-basics/)

## Benefits

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

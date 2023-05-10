% Pixeltable documentation master file, created by
% sphinx-quickstart on Sun Apr 30 19:07:23 2023.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

# Pixeltable: A Table Interface for Image and Video Data

Pixeltable is a Python library that lets computer vision engineers focus on experiments and models
without having to deal with the customary data plumbing.

* Interact with video data at the frame level without having to think about frame extraction,
intermediate file storage, or storage space explosion.
* Augment your data incrementally and interactively with built-in and user-defined functions such as
image transformations, model inference, visualizations, etc., without having to think about data pipelines,
incremental updates, capturing function output, etc.
* Interact with all the data relevant to your CV project (video, images, structured data, JSON) through
a simple dataframe-style API directly in Python. This includes:
    * similarity search on images, supported by high-dimensional vector indexing
    * path expressions and transformations on JSON data
    * PIL and OpenCV image operations
    * assembling frames into videos
* Perform keyword and image similarity search at the video frame level without having to worry about frame
storage.
* Access all Pixeltable-resident data directly as a PyTorch or TensorFlow dataset in your training scripts.
* Understand the compute and storage costs of your data at the granularity of individual augmentations and
get cost projections **before** adding new data and new augmentations.
* Rely on Pixeltable's automatic versioning and snapshot functionality to protect against regressions
and to ensure reproducibility.

## Installation

1. Install Docker

    On MacOS: follow [these](https://docs.docker.com/desktop/install/mac-install/) instructions.

2. `pip install git+https://gitlab.com/pixeltable/python-sdk/`

## Overview

For an overview of Pixeltable, take a look at {doc}`tutorials/Pixeltable Overview`.

:::{toctree}
:hidden:

Home <self>
tutorials/index
api/index
:::
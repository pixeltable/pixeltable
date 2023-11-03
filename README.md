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

1. Using docker (for pg server)
    Install Docker
    On MacOS: follow [these](https://docs.docker.com/desktop/install/mac-install/) instructions.

1b. Using conda or mamba to get pg server, instead of a docker container.
    conda/mamba install postgresql pgvector
    set PIXELTABLE_USE_LOCAL_PG=1 in your env.

2. `pip install git+https://github.com/mkornacker/pixeltable`


## First Steps

[This tutorial](https://pixeltable.readthedocs.io/en/latest/tutorials/Pixeltable%20Overview.html)
gives you a 10-minute overview of Pixeltable.

If you are interested in working with video and how to interact with videos at the frame level through Pixeltable,
take a look at [this tutorial](https://pixeltable.readthedocs.io/en/latest/tutorials/Object%20Detection%20in%20Videos.html).


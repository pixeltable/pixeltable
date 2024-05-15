# Getting Started with Pixeltable

This is a step-by-step guide to setting up a local installation of Pixeltable.

You'll want to install Pixeltable in a Python virtual environment; we'll use Apache Miniconda
in this guide, but any environment manager should work. Pixeltable works with Python 3.9, 3.10,
or 3.11 running on Linux or MacOS.

## Install Pixeltable

1. Install Miniconda here:

    - [Installing Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)

2. Create your environment:

    - `conda create --name pxt python=3.10`
    - `conda activate pxt`

3. Install pixeltable and Jupyter inside the new environment:

    - `pip install pixeltable jupyter`

## Create a Notebook

4. Start your Jupyter notebook server:

    - `jupyter notebook`

5. Create a new notebook:

    - Select "Python 3 (ipykernel)" as the kernel
    - File / New / Notebook

6. Test that everything is working by entering these commands into the notebook:

    - `import pixeltable as pxt`
    - `pxt.list_tables()`

7. Wait a minute for Pixeltable to load; then you should see a message indicating that
    Pixeltable has successfully connected to the database. (The call to `pxt.list_tables()`
    will return an empty list, because you haven't created any tables yet.)

At this point, you're set up to start using Pixeltable! For a tour of what it can
do, a good place to start is the
[Pixeltable Basics](https://pixeltable.github.io/pixeltable/tutorials/pixeltable-basics/)
tutorial.

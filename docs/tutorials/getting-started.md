# Getting Started with Pixeltable

Pixeltable is intended to run on Linux or MacOS.

## Install Pixeltable

You'll want to install Pixeltable in a Python virtual environment. For this tutorial, we'll
use Apache Miniconda, though any environment manager should work.
1. Install Miniconda here: [Installing Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Create your environment:
   - `conda create --name pxt python=3.9`
   - `conda activate pxt`
3. Install pixeltable and Jupyter inside the new environment:
   - `pip install pixeltable jupyter`

## Create a Notebook

4. Start your Jupyter notebook server and create a new notebook:
   - `jupyter notebook`
   - Select "Python 3 (ipykernel)" as the kernel
   - File / New / Notebook
5. Test that everything is working by entering these commands into the notebook:
   - ```
     import pixeltable as pxt
     cl = pxt.Client()
     ```
   
6. Wait a minute for Pixeltable to load; then you should see a message indicating that
   Pixeltable has successfully connected to a database.

At this point, you're set up to start using pixeltable! For a tour of what it can
do, a good place to start is the
[Pixeltable Basics](https://pixeltable.github.io/pixeltable/tutorials/pixeltable-basics/)
tutorial.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Pixeltable'
copyright = '2023, Marcel Kornacker'
author = 'Marcel Kornacker'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for myst-nb -----------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/use/execute.html#configuration
nb_execution_mode = "off"
myst_enable_extensions = [
    "colon_fence",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

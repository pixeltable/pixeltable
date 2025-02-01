from pathlib import Path


def get_templates_path() -> Path:
    """Implementation of the 'mkdocstrings.python.templates' plugin for custom jinja templates."""
    return Path(__file__).parent / 'templates'

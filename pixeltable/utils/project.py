"""Locating the project root, the directory that module references in application files resolve against."""

from __future__ import annotations

from pathlib import Path

import toml

_MARKER = 'pixeltable.toml'
_PYPROJECT = 'pyproject.toml'


def find_project_root(start: Path) -> Path | None:
    """The nearest directory at or above start that marks a project root, or None if no directory does.

    A directory marks a project root by holding a pixeltable.toml, or a pyproject.toml with a
    [tool.pixeltable] section. A directory holding both is a project root by its pixeltable.toml.
    """
    start = start.resolve()
    for dir in (start, *start.parents):
        if (dir / _MARKER).is_file():
            return dir
        pyproject = dir / _PYPROJECT
        if pyproject.is_file() and _declares_pixeltable(pyproject):
            return dir
    return None


def _declares_pixeltable(pyproject: Path) -> bool:
    """True if pyproject holds a [tool.pixeltable] section.

    A file that cannot be read or parsed holds nothing: an unrelated pyproject.toml above the working
    directory decides where a project starts as little as a well-formed one that says nothing about
    Pixeltable.
    """
    try:
        parsed = toml.load(pyproject)
    except Exception:
        return False
    tool = parsed.get('tool')
    return isinstance(tool, dict) and 'pixeltable' in tool

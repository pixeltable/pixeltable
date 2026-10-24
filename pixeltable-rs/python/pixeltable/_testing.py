"""Test hooks used by the compatibility harness (not public API)."""

from pixeltable import _core


def reset_db() -> None:
    """Drop everything in the current database and start with an empty catalog."""
    _core.reset_db()


def reload_catalog() -> None:
    """Drop all in-memory state; the next operation reloads the catalog from disk.

    Table handles held from before keep working: they refer to tables by id.
    """
    _core.reload_catalog()

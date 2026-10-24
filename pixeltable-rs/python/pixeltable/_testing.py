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


def configure_udf_dispatch(threads: int | None = None, async_concurrency: int | None = None) -> tuple[int, int]:
    """Override the number of worker threads for sync UDFs and the default limit of in-flight calls of an async UDF
    (see `PIXELTABLE_UDF_THREADS` and `PIXELTABLE_UDF_ASYNC_CONCURRENCY`) for this process; returns the values in
    effect."""
    return _core.configure_udf_dispatch(threads, async_concurrency)

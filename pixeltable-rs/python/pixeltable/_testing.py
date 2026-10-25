"""Test hooks used by the compatibility harness (not public API)."""

from pixeltable import _core
from pixeltable.func import _runtime


def reset_db() -> None:
    """Drop everything in the current database and start with an empty catalog (and close the idle event loop of
    async UDFs, as upstream's runtime reset does)."""
    _runtime.shutdown(only_if_idle=True)
    _core.reset_db()


def reload_catalog() -> None:
    """Drop all in-memory state; the next operation reloads the catalog from disk (and closes the idle event loop of
    async UDFs; the next async call starts a new one).

    Table handles held from before keep working: they refer to tables by id.
    """
    _runtime.shutdown(only_if_idle=True)
    _core.reload_catalog()


def validate_store() -> None:
    """Check that the store is consistent and matches the in-memory catalog (upstream validates its store after each
    test): every table's files exist and no file is unreferenced, row version ranges are well formed, at most one live
    row per key, primary-key and B-tree index values match their columns, and the catalog reloaded from disk equals
    the one in memory. Raises AssertionError listing the problems."""
    problems = _core.validate_store()
    if problems:
        raise AssertionError('store validation failed:\n  ' + '\n  '.join(problems))


def configure_udf_dispatch(threads: int | None = None, async_concurrency: int | None = None) -> tuple[int, int]:
    """Override the number of worker threads for sync UDFs and the default limit of in-flight calls of an async UDF
    (see `PIXELTABLE_UDF_THREADS` and `PIXELTABLE_UDF_ASYNC_CONCURRENCY`) for this process; returns the values in
    effect."""
    return _core.configure_udf_dispatch(threads, async_concurrency)

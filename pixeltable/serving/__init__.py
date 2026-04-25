"""
Adapters for web serving frameworks.
"""

from typing import Any

try:
    from ._fastapi import FastAPIRouter, SqlExport
except ImportError:
    # fastapi is an optional dependency; provide a stub that raises a helpful error on use
    class FastAPIRouter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "pixeltable.serving.FastAPIRouter requires fastapi; install it with `pip install 'fastapi[standard]'`"
            )

    class SqlExport:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "pixeltable.serving.SqlExport requires fastapi; install it with `pip install 'fastapi[standard]'`"
            )


__all__ = ['FastAPIRouter', 'SqlExport']

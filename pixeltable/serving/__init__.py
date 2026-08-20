"""
Adapters for web serving frameworks.
"""

from typing import Any

from pixeltable.config import SqlExport

from ._diff import ServiceChangeOp, Severity
from ._spec import RouteSpec, ServiceSpec

_NO_FASTAPI_MSG = "pixeltable.serving.FastAPIRouter requires fastapi; install it with `pip install 'pixeltable[serve]'`"

# the methods an application file calls to declare its routes; mirrored by the stub below, and checked
# against the real router by tests/serving/test_stub_router.py
ROUTE_DECLARATION_METHODS = (
    'add_compute_route',
    'add_delete_route',
    'add_insert_route',
    'add_query_route',
    'add_update_route',
    'compute_route',
    'insert_route',
    'update_route',
)

try:
    from ._fastapi import FastAPIRouter
except ImportError:
    # fastapi is an optional dependency; provide a stub that records nothing and reports what is missing
    class FastAPIRouter:  # type: ignore[no-redef]
        """Stand-in for the router when fastapi is not installed.

        An application file declares its routes when it is imported, and importing that file is how schema
        operations reach the models it declares, so declaring a route succeeds here and records nothing.
        Serving needs fastapi, so everything else reports that it is missing.
        """

        name: str | None
        prefix: str

        def __init__(self, *args: Any, name: str | None = None, prefix: str = '', **kwargs: Any) -> None:
            self.name = name
            self.prefix = prefix

        def __getattr__(self, attr_name: str) -> Any:
            if attr_name not in ROUTE_DECLARATION_METHODS:
                raise ImportError(_NO_FASTAPI_MSG)

            def declare(*args: Any, **kwargs: Any) -> Any:
                # the decorator forms return the decorated function unchanged; the add_*_route forms ignore
                # what they get back
                return lambda fn: fn

            return declare


__all__ = [
    'ROUTE_DECLARATION_METHODS',
    'FastAPIRouter',
    'RouteSpec',
    'ServiceChangeOp',
    'ServiceSpec',
    'Severity',
    'SqlExport',
]

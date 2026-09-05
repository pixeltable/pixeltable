"""
Adapters for web serving frameworks.
"""

from typing import Any

from pixeltable.serving.globals import SqlExport

from .service_instance import ServiceInstance, ServiceInstanceRecord, ServiceInstanceState

_NO_FASTAPI_MSG = "pixeltable.serving.FastAPIRouter requires fastapi; install it with `pip install 'pixeltable[serve]'`"

# the methods an application file calls to define its routes; mirrored by the stub below, and checked
# against the real router by tests/serving/test_stub_router.py
ROUTE_DEFINITION_METHODS = (
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

        An application file defines its routes when it is imported, and importing that file is how schema
        operations reach the models it defines, so defining a route succeeds here and records nothing.
        Serving needs fastapi, so everything else reports that it is missing.
        """

        name: str | None
        prefix: str

        def __init__(self, *args: Any, name: str | None = None, prefix: str = '', **kwargs: Any) -> None:
            self.name = name
            self.prefix = prefix

        def __getattr__(self, attr_name: str) -> Any:
            if attr_name not in ROUTE_DEFINITION_METHODS:
                raise ImportError(_NO_FASTAPI_MSG)

            def define(*args: Any, **kwargs: Any) -> Any:
                # the decorator forms return the decorated function unchanged; the add_*_route forms ignore
                # what they get back
                return lambda fn: fn

            return define


__all__ = [
    'ROUTE_DEFINITION_METHODS',
    'FastAPIRouter',
    'ServiceInstance',
    'ServiceInstanceRecord',
    'ServiceInstanceState',
    'SqlExport',
]

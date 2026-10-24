"""Query template functions (`@pxt.query`, `pxt.retrieval_udf()`): placeholder until they land in the backend."""

from __future__ import annotations

from typing import Any, Callable

from ._function import Function


class QueryTemplateFunction(Function):
    """A function defined by a query template (`@pxt.query`)."""

    def __init__(self, py_fn: Callable, **options: Any):
        super().__init__([], f'{py_fn.__module__}.{py_fn.__qualname__}')
        self.py_fn = py_fn
        self.options = options

    @property
    def name(self) -> str:
        return self.py_fn.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f'calling the query function `{self.name}` is not implemented yet in pixeltable-rs')


def query(*args: Any, **kwargs: Any) -> Any:
    """Decorator for query template functions."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return QueryTemplateFunction(args[0])
    return lambda fn: QueryTemplateFunction(fn, **kwargs)


def retrieval_udf(table: Any, name: str | None = None, description: str | None = None, **kwargs: Any) -> Function:
    raise NotImplementedError('retrieval_udf() is not implemented yet in pixeltable-rs')

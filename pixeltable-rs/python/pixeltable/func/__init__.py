"""Functions, UDFs, UDAs and iterators.

Placeholder implementation: the decorators (`pxt.udf`, `pxt.uda`, `pxt.expr_udf`, `pxt.query`, `pxt.iterator`)
create function objects at import time, so that modules defining UDFs import cleanly, but calling a UDF to build an
expression raises NotImplementedError until Python UDF support lands. The backend's extension point is
`functions::FuncRef::Udf` (see ARCHITECTURE.md).
"""

from __future__ import annotations

import abc
import typing
from typing import Any, Callable, Generic, Iterator, Self, TypeVar

import pydantic

T = TypeVar('T')

Batch = typing.Annotated[list[T], 'pxt-batch']


def _not_implemented(what: str) -> NotImplementedError:
    return NotImplementedError(f'{what} is not implemented yet in pixeltable-rs')


class Function(abc.ABC):
    """Base class of Pixeltable functions."""

    name: str
    self_path: str | None

    def __init__(self, name: str, self_path: str | None = None):
        self.name = name
        self.self_path = self_path
        self._overloads: list[Callable] = []

    @property
    def display_name(self) -> str:
        return self.name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise _not_implemented(f'calling the function `{self.display_name}`')

    # decorator hooks used when defining functions; recorded for the UDF implementation
    def overload(self, fn: Callable) -> Self:
        self._overloads.append(fn)
        return self

    def conditional_return_type(self, fn: Callable) -> Callable:
        self._conditional_return_type = fn
        return fn

    def resource_estimator(self, fn: Callable) -> Callable:
        self._resource_estimator = fn
        return fn

    def using(self, **kwargs: Any) -> Self:
        raise _not_implemented('Function.using()')

    def to_sql(self, fn: Callable) -> Callable:
        return fn

    def __repr__(self) -> str:
        return f'<Pixeltable function {self.self_path or self.name}>'


class InvalidFunction(Function):
    pass


class CallableFunction(Function):
    """A function implemented by a Python callable (`@pxt.udf`)."""

    def __init__(self, py_fn: Callable, *, batch_size: int | None = None, **options: Any):
        super().__init__(py_fn.__name__, f'{py_fn.__module__}.{py_fn.__qualname__}')
        self.py_fn = py_fn
        self.batch_size = batch_size
        self.options = options


class ExprTemplateFunction(Function):
    """A function defined by an expression template (`@pxt.expr_udf`)."""

    def __init__(self, py_fn: Callable, **options: Any):
        super().__init__(py_fn.__name__, f'{py_fn.__module__}.{py_fn.__qualname__}')
        self.py_fn = py_fn


class QueryTemplateFunction(Function):
    """A function defined by a query template (`@pxt.query`)."""

    def __init__(self, py_fn: Callable, **options: Any):
        super().__init__(py_fn.__name__, f'{py_fn.__module__}.{py_fn.__qualname__}')
        self.py_fn = py_fn


class Aggregator(abc.ABC):
    """Base class of user-defined aggregators (`@pxt.uda`)."""

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def value(self) -> Any: ...


class AggregateFunction(Function):
    def __init__(self, cls: type, **options: Any):
        super().__init__(cls.__name__, f'{cls.__module__}.{cls.__qualname__}')
        self.agg_class = cls
        self.options = options


class PxtIterator(abc.ABC, Iterator[T], Generic[T]):
    """Base class of class-based iterators (`@pxt.iterator`)."""

    def __iter__(self) -> Self:
        return self

    @abc.abstractmethod
    def __next__(self) -> T: ...


class GeneratingFunction(Function):
    """An iterator function (`@pxt.iterator`)."""

    def __init__(self, decorated_callable: Callable, unstored_cols: list[str] | None = None):
        name = getattr(decorated_callable, '__name__', type(decorated_callable).__name__)
        super().__init__(name, f'{decorated_callable.__module__}.{getattr(decorated_callable, "__qualname__", name)}')
        self.decorated_callable = decorated_callable
        self.unstored_cols = unstored_cols or []

    def validate(self, fn: Callable) -> Callable:
        self._validate = fn
        return fn

    def conditional_output_schema(self, fn: Callable) -> Callable:
        self._conditional_output_schema = fn
        return fn


class GeneratingFunctionCall:
    """A call of an iterator function (not implemented yet)."""


class FunctionRegistry:
    """Registry of functions (not implemented yet)."""

    @classmethod
    def get(cls) -> FunctionRegistry:
        raise _not_implemented('FunctionRegistry')


class Parameter:
    pass


class Signature:
    pass


def _decorator(make: Callable[..., Function], args: tuple, kwargs: dict) -> Any:
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return make(args[0])
    return lambda fn: make(fn, **kwargs)


def udf(*args: Any, **kwargs: Any) -> Any:
    """Decorator for user-defined functions: `@pxt.udf` or `@pxt.udf(batch_size=...)`."""
    return _decorator(CallableFunction, args, kwargs)


def make_function(decorated_fn: Callable, **kwargs: Any) -> Function:
    return CallableFunction(decorated_fn, **kwargs)


def expr_udf(*args: Any, **kwargs: Any) -> Any:
    """Decorator for functions defined by an expression template."""
    return _decorator(ExprTemplateFunction, args, kwargs)


def uda(*args: Any, **kwargs: Any) -> Any:
    """Decorator for user-defined aggregate functions (on a subclass of `Aggregator`)."""
    return _decorator(AggregateFunction, args, kwargs)


def query(*args: Any, **kwargs: Any) -> Any:
    """Decorator for query template functions."""
    return _decorator(QueryTemplateFunction, args, kwargs)


def iterator(*args: Any, **kwargs: Any) -> Any:
    """Decorator for iterator functions (generator functions or `PxtIterator` subclasses)."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return GeneratingFunction(args[0])
    return lambda fn: GeneratingFunction(fn, **kwargs)


def retrieval_udf(table: Any, name: str | None = None, description: str | None = None, **kwargs: Any) -> Function:
    raise _not_implemented('retrieval_udf()')


def mcp_udfs(url: str) -> list[Function]:
    raise _not_implemented('mcp_udfs()')


class Tool(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    fn: Any
    name: str | None = None
    description: str | None = None


class ToolChoice(pydantic.BaseModel):
    auto: bool
    required: bool
    tool: str | None
    parallel_tool_calls: bool


class Tools(pydantic.BaseModel):
    tools: list[Tool]

"""Iterators (`@pxt.iterator`): placeholder until iterator calls land in the backend."""

from __future__ import annotations

import abc
from typing import Any, Callable, Generic, Iterator, Self, TypeVar

T = TypeVar('T')


class PxtIterator(abc.ABC, Iterator[T], Generic[T]):
    """Base class of class-based iterators (`@pxt.iterator`)."""

    def __iter__(self) -> Self:
        return self

    @abc.abstractmethod
    def __next__(self) -> T: ...


class GeneratingFunction:
    """An iterator function (`@pxt.iterator`)."""

    def __init__(self, decorated_callable: Callable, unstored_cols: list[str] | None = None, fqn: str | None = None):
        self.decorated_callable = decorated_callable
        self.unstored_cols = unstored_cols or []
        name = getattr(decorated_callable, '__name__', type(decorated_callable).__name__)
        self.fqn = fqn or f'{decorated_callable.__module__}.{getattr(decorated_callable, "__qualname__", name)}'

    @property
    def name(self) -> str:
        return self.fqn.rsplit('.', 1)[-1]

    def validate(self, fn: Callable) -> Callable:
        self._validate = fn
        return fn

    def conditional_output_schema(self, fn: Callable) -> Callable:
        self._conditional_output_schema = fn
        return fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f'calling the iterator `{self.name}` is not implemented yet in pixeltable-rs')


class GeneratingFunctionCall:
    """A call of an iterator function (not implemented yet)."""


def iterator(*args: Any, **kwargs: Any) -> Any:
    """Decorator for iterator functions (generator functions or `PxtIterator` subclasses)."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return GeneratingFunction(args[0])
    return lambda fn: GeneratingFunction(fn, **kwargs)

"""User-defined aggregate functions (`@pxt.uda`): placeholder until aggregation lands in the backend."""

from __future__ import annotations

import abc
from typing import Any

from ._function import Function


class Aggregator(abc.ABC):
    """Base class of user-defined aggregators (`@pxt.uda`)."""

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def value(self) -> Any: ...


class AggregateFunction(Function):
    def __init__(self, cls: type, **options: Any):
        super().__init__([], f'{cls.__module__}.{cls.__qualname__}')
        self.agg_class = cls
        self.options = options
        self.__doc__ = cls.__doc__
        self._overload_classes: list[type] = [cls]

    @property
    def name(self) -> str:
        return self.agg_class.__name__

    def overload(self, cls: type) -> AggregateFunction:
        self._overload_classes.append(cls)
        return self

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(f'calling the aggregate function `{self.name}` is not implemented yet in pixeltable-rs')


def uda(*args: Any, **kwargs: Any) -> Any:
    """Decorator for user-defined aggregate functions (on a subclass of `Aggregator`)."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return AggregateFunction(args[0])
    return lambda cls: AggregateFunction(cls, **kwargs)

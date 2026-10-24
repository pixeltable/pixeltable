"""Helpers for public API that exists but is not implemented yet: calling it raises NotImplementedError."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar('F', bound=Callable[..., Any])


def _message(what: str) -> str:
    return f'{what} is not implemented yet in pixeltable-rs'


def unimplemented(what: str) -> Callable[[F], F]:
    """Decorator: keep the signature and docstring, raise NotImplementedError when called."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError(_message(what))

        return wrapper  # type: ignore[return-value]

    return decorator


class UnimplementedFunction:
    """Placeholder for a public function (e.g. a UDF in `pixeltable.functions`) that is not implemented yet."""

    def __init__(self, module: str, name: str):
        self.__module__ = module
        self.__name__ = name
        self.__qualname__ = name
        self.name = name
        self.self_path = f'{module}.{name}'

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_message(f'`{self.self_path}()`'))

    def __repr__(self) -> str:
        return f'<unimplemented function {self.self_path}>'


def stub_functions(module_name: str, names: list[str]) -> dict[str, UnimplementedFunction]:
    return {name: UnimplementedFunction(module_name, name) for name in names}


def unimplemented_class(module: str, name: str) -> type:
    """A placeholder class whose instantiation raises NotImplementedError."""

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_message(f'`{module}.{name}`'))

    @classmethod  # type: ignore[misc]
    def create(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(_message(f'`{module}.{name}`'))

    return type(name, (), {'__init__': __init__, '__module__': module, 'create': create})

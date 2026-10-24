"""Resolution of `pixeltable.functions` attributes to backend builtins (or placeholders)."""

from __future__ import annotations

from typing import Any

from pixeltable import _core
from pixeltable._unimplemented import UnimplementedFunction
from pixeltable.func import Function

_BUILTINS: set[str] = set(_core.builtin_names())


class BuiltinFunction(Function):
    """A function implemented in the Rust backend; calling it builds a `FunctionCall` expression."""

    def __init__(self, module: str, name: str):
        super().__init__([], f'{module}.{name}')
        self.__module__ = module
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> _core.Expr:
        return _core.call_builtin(self.self_path, *args, **kwargs)


def fn(module: str, name: str) -> Function | UnimplementedFunction:
    """The builtin `module.name` if the backend implements it, else a placeholder that raises when called."""
    if f'{module}.{name}' in _BUILTINS:
        return BuiltinFunction(module, name)
    return UnimplementedFunction(module, name)

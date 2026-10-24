"""Resolution of `pixeltable.functions` attributes to backend builtins (or placeholders)."""

from __future__ import annotations

from typing import Any, Callable

from pixeltable import _core
from pixeltable._unimplemented import UnimplementedFunction
from pixeltable.func import Function

_BUILTINS: set[str] = set(_core.builtin_names())


class BuiltinFunction(Function):
    """A function implemented in the Rust backend; calling it builds a `FunctionCall` expression."""

    def __init__(self, module: str, name: str):
        super().__init__(name, f'{module}.{name}')
        self.__module__ = module
        self.__name__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> _core.Expr:
        return _core.call_builtin(self.self_path, *args, **kwargs)


def _map(expr: Any, fn: Callable[[Any], Any]) -> _core.Expr:
    """Applies a function to each element of a JSON array, producing a new array."""
    return _core.json_mapper('map', expr, fn)


def _filter(expr: Any, predicate: Callable[[Any], Any]) -> _core.Expr:
    """Keeps the elements of a JSON array for which a predicate holds, producing a new array."""
    return _core.json_mapper('filter', expr, predicate)


def _sort(expr: Any, key: Callable[[Any], Any] | None = None, *, asc: bool = True) -> _core.Expr:
    """Sorts the elements of a JSON array, producing a new array."""
    return _core.json_mapper('sort', expr, key, asc)


def _cast(expr: Any, target_type: Any) -> _core.Expr:
    """The same expression with another static type (no conversion of values)."""
    return _core.cast(expr, target_type)


# functions that take Python callables or types rather than expressions
_SPECIAL: dict[str, Callable[..., Any]] = {
    'pixeltable.functions.globals.map': _map,
    'pixeltable.functions.globals.filter': _filter,
    'pixeltable.functions.globals.sort': _sort,
    'pixeltable.functions.globals.cast': _cast,
}


def fn(module: str, name: str) -> Function | UnimplementedFunction | Callable[..., Any]:
    """The builtin `module.name` if the backend implements it, else a placeholder that raises when called."""
    qualified = f'{module}.{name}'
    if qualified in _SPECIAL:
        special = _SPECIAL[qualified]
        special.__module__ = module
        special.__name__ = name
        special.__qualname__ = name
        return special
    if qualified in _BUILTINS:
        return BuiltinFunction(module, name)
    return UnimplementedFunction(module, name)

from typing import List, Callable
import importlib

import pixeltable.type_system as ts


def resolve_symbol(module_name: str, symbol: str) -> object:
    module = importlib.import_module(module_name)
    obj = module
    for el in symbol.split('.'):
        obj = getattr(obj, el)
    return obj

def udf(*, return_type: ts.ColumnType, param_types: List[ts.ColumnType]) -> Callable:
    """Returns decorator to create a Function from a function definition.

    Example:
        >>> @pt.udf(param_types=[pt.IntType()], return_type=pt.IntType())
        ... def my_function(x):
        ...    return x + 1
    """
    from .function import Function
    def decorator(fn: Callable) -> Function:
        return Function.make_function(return_type, param_types, fn)
    return decorator

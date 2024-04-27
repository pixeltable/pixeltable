import importlib
import inspect
from types import ModuleType
from typing import Optional

import pixeltable.exceptions as excs


def resolve_symbol(symbol_path: str) -> Optional[object]:
    path_elems = symbol_path.split('.')
    module: Optional[ModuleType] = None
    i = len(path_elems) - 1
    while i > 0 and module is None:
        try:
            module = importlib.import_module('.'.join(path_elems[:i]))
        except ModuleNotFoundError:
            i -= 1
    if i == 0:
        return None  # Not resolvable
    obj = module
    for el in path_elems[i:]:
        obj = getattr(obj, el)
    return obj


def validate_symbol_path(fn_path: str) -> None:
    path_elems = fn_path.split('.')
    fn_name = path_elems[-1]
    if any(el == '<locals>' for el in path_elems):
        raise excs.Error(
            f'{fn_name}(): nested functions are not supported. Move the function to the module level or into a class.')
    if any(not el.isidentifier() for el in path_elems):
        raise excs.Error(
            f'{fn_name}(): cannot resolve symbol path {fn_path}. Move the function to the module level or into a class.')


def get_caller_module_path() -> str:
    """Return the module path of our caller's caller"""
    stack = inspect.stack()
    try:
        caller_frame = stack[2].frame
        module_path = caller_frame.f_globals['__name__']
    finally:
        # remove references to stack frames to avoid reference cycles
        del stack
    return module_path

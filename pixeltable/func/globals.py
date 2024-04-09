from typing import Optional
from types import ModuleType
import importlib
import inspect


def resolve_symbol(symbol_path: str) -> object:
    path_elems = symbol_path.split('.')
    module: Optional[ModuleType] = None
    if path_elems[0:2] == ['pixeltable', 'functions'] and len(path_elems) > 2:
        # if this is a pixeltable.functions submodule, it cannot be resolved via pixeltable.functions;
        # try to import the submodule directly
        submodule_path = '.'.join(path_elems[0:3])
        try:
            module = importlib.import_module(submodule_path)
            path_elems = path_elems[3:]
        except ModuleNotFoundError:
            pass
    if module is None:
        module = importlib.import_module(path_elems[0])
        path_elems = path_elems[1:]
    obj = module
    for el in path_elems:
        obj = getattr(obj, el)
    return obj

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

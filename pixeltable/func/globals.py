import importlib
import inspect


def resolve_symbol(symbol_path: str) -> object:
    path_elems = symbol_path.split('.')
    module = importlib.import_module(path_elems[0])
    obj = module
    for el in path_elems[1:]:
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

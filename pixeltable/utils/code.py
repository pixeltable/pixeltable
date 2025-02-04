import types
from typing import Optional

from pixeltable.func import Function

# Utilities related to the organization of the Pixeltable codebase.


def local_public_names(mod_name: str, exclude: Optional[list[str]] = None) -> list[str]:
    """
    Returns a list of all functions and submodules that are local to the specified module and are
    publicly accessible. Intended to facilitate implementation of module __dir__() methods for
    friendly tab-completion.
    """
    import importlib

    if exclude is None:
        exclude = []
    mod = importlib.import_module(mod_name)
    names = []
    for obj in mod.__dict__.values():
        if isinstance(obj, Function):
            # Pixeltable function
            names.append(obj.name)
        elif isinstance(obj, types.FunctionType):
            # Python function
            if obj.__module__ == mod.__name__ and not obj.__name__.startswith('_'):
                names.append(obj.__name__)
        elif isinstance(obj, types.ModuleType):
            # Module
            components = obj.__name__.split('.')
            if mod_name == '.'.join(components[:-1]):
                names.append(components[-1])
    return [name for name in names if name not in exclude]

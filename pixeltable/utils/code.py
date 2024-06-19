import types
from typing import Optional

from pixeltable.func import Function


def local_public_names(mod_name: str, exclude: Optional[list[str]] = None) -> list[str]:
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

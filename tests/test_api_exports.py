import importlib
import pkgutil
from types import FunctionType
from typing import Any

import pytest

import pixeltable as pxt
from pixeltable.func import Function, GeneratingFunction


@pytest.mark.parametrize(
    'module_name',
    [
        'pixeltable',
        'pixeltable.io',
        'pixeltable.functions',
        *sorted(info.name for info in pkgutil.walk_packages(pxt.functions.__path__, 'pixeltable.functions.')),
    ],
)
def test_public_exports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, '__all__'):
        return
    parent_name, _, name = module_name.rpartition('.')
    if parent_name == 'pixeltable.functions' and name != 'globals':
        assert name in pxt.functions.__all__, f'Add {name!r} to __all__ in pixeltable/functions/__init__.py.'
    duplicates = sorted({name for name in module.__all__ if module.__all__.count(name) > 1})
    assert not duplicates, f'Remove duplicate names {duplicates} from {module_name}.__all__.'
    undefined = [name for name in module.__all__ if not hasattr(module, name)]
    assert not undefined, f'Remove undefined names {undefined} from {module_name}.__all__.'
    for name, value in vars(module).items():
        if name.startswith('_'):
            continue
        if isinstance(value, Function):
            defined_module = value.self_path.rpartition('.')[0] if value.self_path is not None else None
        elif isinstance(value, GeneratingFunction):
            defined_module = value.fqn.rpartition('.')[0]
        elif isinstance(value, FunctionType):
            defined_module = value.__module__
        else:
            continue
        if defined_module == module_name:
            assert name in module.__all__, f'{name!r} is public in {module_name}; add it to {module_name}.__all__.'

    if module_name == 'pixeltable.functions.video':
        children = ('editing', 'filters', 'iterators', 'scene_detect')
        child_exports = set().union(*(importlib.import_module(f'{module_name}.{child}').__all__ for child in children))
        missing = sorted(child_exports - set(module.__all__))
        extra = sorted(set(module.__all__) - child_exports)
        assert not missing, f'Import {missing} in pixeltable/functions/video/__init__.py and add them to its __all__.'
        assert not extra, f'Add {extra} to the __all__ of the video submodule that defines them.'

    namespace: dict[str, Any] = {}
    exec(f'from {module_name} import *', namespace)
    for name in module.__all__:
        assert namespace[name] is getattr(module, name)
    if '__dir__' in vars(module):
        assert dir(module) == sorted(module.__all__)

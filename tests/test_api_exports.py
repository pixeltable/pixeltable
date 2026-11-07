import importlib
import pkgutil
from typing import Any

import pytest

import pixeltable as pxt


@pytest.mark.parametrize(
    'module_name',
    [
        'pixeltable',
        'pixeltable.io',
        'pixeltable.iterators',
        'pixeltable.functions',
        *sorted(info.name for info in pkgutil.walk_packages(pxt.functions.__path__, 'pixeltable.functions.')),
    ],
)
def test_public_exports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, '__all__'):
        return
    assert len(module.__all__) == len(set(module.__all__))
    namespace: dict[str, Any] = {}
    exec(f'from {module_name} import *', namespace)
    assert set(namespace) - {'__builtins__'} == set(module.__all__)
    for name in module.__all__:
        assert namespace[name] is getattr(module, name)
    if '__dir__' in vars(module):
        assert dir(module) == sorted(module.__all__)

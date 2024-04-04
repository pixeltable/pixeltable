from __future__ import annotations

import inspect
import sys
from typing import Optional, Dict, Callable, List, Tuple
from uuid import UUID
import cloudpickle

import pixeltable.type_system as ts
import pixeltable.exceptions as excs
from .function import Function
from .function_registry import FunctionRegistry
from .globals import get_caller_module_path
from .signature import Signature


class CallableFunction(Function):
    """Pixeltable Function backed by a Python Callable.

    CallableFunctions come in two flavors:
    - references to lambdas and functions defined in notebooks, which are pickled and serialized to the store
    - functions that are defined in modules are serialized via the default mechanism
    """

    def __init__(
            self, signature: Signature, py_fn: Callable, self_path: Optional[str] = None,
            self_name: Optional[str] = None):
        assert py_fn is not None
        self.py_fn = py_fn
        self.self_name = self_name
        py_signature = inspect.signature(self.py_fn)
        super().__init__(signature, py_signature, self_path=self_path)

    @property
    def display_name(self) -> str:
        return self.self_name

    @property
    def name(self) -> str:
        return self.self_name

    def help_str(self) -> str:
        res = super().help_str()
        res += '\n\n' + inspect.getdoc(self.py_fn)
        return res

    def _as_dict(self) -> Dict:
        if self.self_path is None:
            # this is not a module function
            from .function_registry import FunctionRegistry
            id = FunctionRegistry.get().create_stored_function(self)
            return {'id': id.hex}
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, d: Dict) -> Function:
        if 'id' in d:
            from .function_registry import FunctionRegistry
            return FunctionRegistry.get().get_stored_function(UUID(hex=d['id']))
        return super()._from_dict(d)

    def to_store(self) -> Tuple[Dict, bytes]:
        return (self.signature.as_dict(), cloudpickle.dumps(self.py_fn))

    @classmethod
    def from_store(cls, name: Optional[str], md: Dict, binary_obj: bytes) -> Function:
        py_fn = cloudpickle.loads(binary_obj)
        assert isinstance(py_fn, Callable)
        return CallableFunction(Signature.from_dict(md), py_fn, self_name=name)

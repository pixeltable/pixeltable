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


def make_callable_function(
        py_fn: Callable, *, return_type: ts.ColumnType, param_types: List[ts.ColumnType],
        function_path: Optional[str] = None, function_name: Optional[str] = None
) -> CallableFunction:
    """
    Args:
        function_path: path of the returned CallableFunction
        function_name: name of the CallableFunction
    """
    sig = Signature.create(py_fn, param_types, return_type)

    if function_path is None:
        # this is not a named function in a module;
        # we preserve the name
        return CallableFunction(sig, py_fn=py_fn, self_name=function_name or py_fn.__name__)

    result = CallableFunction(sig, py_fn=py_fn, self_path=function_path, self_name=function_name)
    FunctionRegistry.get().register_function(function_path, result)
    return result


def udf(*, return_type: ts.ColumnType, param_types: List[ts.ColumnType], py_fn: Optional[Callable] = None) -> Callable:
    """Returns decorator to create a CallableFunction from a function definition.

    Example:
        >>> @pxt.udf(param_types=[pt.IntType()], return_type=pt.IntType())
        ... def my_function(x):
        ...    return x + 1
    """
    def decorator(py_fn: Callable) -> CallableFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            # this is a named function in a module
            function_path = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None
        return make_callable_function(
            py_fn, return_type=return_type, param_types=param_types,
            function_path=function_path, function_name=py_fn.__name__)

    # the decorated function is only used for the signature/path and never executed
    def dummy_decorator(dummy_py_fn: Callable) -> CallableFunction:
        if dummy_py_fn.__module__ == '__main__':
            raise excs.Error('The @udf decorator with the explicit py_fn argument can only be used in a module')
        return make_callable_function(
            py_fn, return_type=return_type, param_types=param_types,
            function_path=f'{dummy_py_fn.__module__}.{dummy_py_fn.__qualname__}', function_name=dummy_py_fn.__name__)

    return decorator if py_fn is None else dummy_decorator

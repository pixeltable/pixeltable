from __future__ import annotations

import inspect
from typing import Optional, Callable, Tuple, Any
from uuid import UUID

import cloudpickle

from .function import Function
from .signature import Signature


class CallableFunction(Function):
    """Pixeltable Function backed by a Python Callable.

    CallableFunctions come in two flavors:
    - references to lambdas and functions defined in notebooks, which are pickled and serialized to the store
    - functions that are defined in modules are serialized via the default mechanism
    """

    def __init__(
            self, signature: Signature, py_fn: Callable, self_path: Optional[str] = None,
            self_name: Optional[str] = None, batch_size: Optional[int] = None):
        assert py_fn is not None
        self.py_fn = py_fn
        self.self_name = self_name
        self.batch_size = batch_size
        self.__doc__ = py_fn.__doc__
        super().__init__(signature, self_path=self_path)

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None

    def exec(self, *args: Any, **kwargs: Any) -> Any:
        if self.is_batched:
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result = self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        else:
            return self.py_fn(*args, **kwargs)

    def exec_batch(self, *args: Any, **kwargs: Any) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        # Unpack the constant parameters
        constant_param_names = [p.name for p in self.signature.constant_parameters]
        constant_kwargs = {k: v[0] for k, v in kwargs.items() if k in constant_param_names}
        batched_kwargs = {k: v for k, v in kwargs.items() if k not in constant_param_names}
        return self.py_fn(*args, **constant_kwargs, **batched_kwargs)

    # TODO(aaron-siegel): Implement conditional batch sizing
    def get_batch_size(self, *args: Any, **kwargs: Any) -> Optional[int]:
        return self.batch_size

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

    def _as_dict(self) -> dict:
        if self.self_path is None:
            # this is not a module function
            from .function_registry import FunctionRegistry
            id = FunctionRegistry.get().create_stored_function(self)
            return {'id': id.hex}
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'id' in d:
            from .function_registry import FunctionRegistry
            return FunctionRegistry.get().get_stored_function(UUID(hex=d['id']))
        return super()._from_dict(d)

    def to_store(self) -> tuple[dict, bytes]:
        md = {
            'signature': self.signature.as_dict(),
            'batch_size': self.batch_size,
        }
        return md, cloudpickle.dumps(self.py_fn)

    @classmethod
    def from_store(cls, name: Optional[str], md: dict, binary_obj: bytes) -> Function:
        py_fn = cloudpickle.loads(binary_obj)
        assert isinstance(py_fn, Callable)
        sig = Signature.from_dict(md['signature'])
        batch_size = md['batch_size']
        return CallableFunction(sig, py_fn, self_name=name, batch_size=batch_size)

    def validate_call(self, bound_args: dict[str, Any]) -> None:
        import pixeltable.exprs as exprs
        if self.is_batched:
            for param in self.signature.constant_parameters:
                if param.name in bound_args and isinstance(bound_args[param.name], exprs.Expr):
                    raise ValueError(
                        f'{self.display_name}(): '
                        f'parameter {param.name} must be a constant value, not a Pixeltable expression'
                    )

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'

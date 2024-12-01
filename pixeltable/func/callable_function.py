from __future__ import annotations

import inspect
from typing import Any, Callable, Iterable, Optional
from uuid import UUID

import cloudpickle  # type: ignore[import-untyped]

from .function import Function
from .signature import Signature


class CallableFunction(Function):
    """Pixeltable Function backed by a Python Callable.

    CallableFunctions come in two flavors:
    - references to lambdas and functions defined in notebooks, which are pickled and serialized to the store
    - functions that are defined in modules are serialized via the default mechanism
    """

    def __init__(
        self,
        signatures: list[Signature],
        py_fns: list[Callable],
        self_path: Optional[str] = None,
        self_name: Optional[str] = None,
        batch_size: Optional[int] = None,
        is_method: bool = False,
        is_property: bool = False
    ):
        assert len(py_fns) > 0
        self.py_fns = py_fns
        self.self_name = self_name
        self.batch_size = batch_size
        self.__doc__ = py_fns[0].__doc__
        super().__init__(signatures, self_path=self_path, is_method=is_method, is_property=is_property)

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None

    def exec(self, sig_idx: int, args: list[Any], kwargs: dict[str, Any]) -> Any:
        signature = self.signatures[sig_idx]
        py_fn = self.py_fns[sig_idx]
        if self.is_batched:
            assert signature in self.signatures
            # Pack the batched parameters into singleton lists
            constant_param_names = [p.name for p in signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result = py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            assert len(result) == 1
            return result[0]
        else:
            return py_fn(*args, **kwargs)

    def exec_batch(self, sig_idx: int, args: list[Any], **kwargs: dict[str, Any]) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        signature = self.signatures[sig_idx]
        py_fn = self.py_fns[sig_idx]
        # Unpack the constant parameters
        constant_param_names = [p.name for p in signature.constant_parameters]
        constant_kwargs = {k: v[0] for k, v in kwargs.items() if k in constant_param_names}
        batched_kwargs = {k: v for k, v in kwargs.items() if k not in constant_param_names}
        return py_fn(*args, **constant_kwargs, **batched_kwargs)

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
            assert not self.is_method and not self.is_property
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
            'signatures': [signature.as_dict() for signature in self.signatures],
            'batch_size': self.batch_size,
        }
        return md, cloudpickle.dumps(self.py_fns)

    @classmethod
    def from_store(cls, name: Optional[str], md: dict, binary_obj: bytes) -> Function:
        py_fns = cloudpickle.loads(binary_obj)
        assert isinstance(py_fns, list)
        assert all(callable(fn) for fn in py_fns)
        sigs = [Signature.from_dict(sig_md) for sig_md in md['signatures']]
        batch_size = md['batch_size']
        return CallableFunction(sigs, py_fns, self_name=name, batch_size=batch_size)

    def validate_call(self, signature: Signature, bound_args: dict[str, Any]) -> None:
        import pixeltable.exprs as exprs
        if self.is_batched:
            for param in signature.constant_parameters:
                if param.name in bound_args and isinstance(bound_args[param.name], exprs.Expr):
                    raise ValueError(
                        f'{self.display_name}(): '
                        f'parameter {param.name} must be a constant value, not a Pixeltable expression'
                    )

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'

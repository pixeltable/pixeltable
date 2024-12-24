from __future__ import annotations

import inspect
from typing import Any, Callable, Optional, Sequence
from uuid import UUID

import cloudpickle  # type: ignore[import-untyped]

import pixeltable.exceptions as excs

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
        assert len(signatures) > 0
        assert len(signatures) == len(py_fns)
        if self_path is None and len(signatures) > 1:
            raise excs.Error('Multiple signatures are only allowed for module UDFs (not locally defined UDFs)')
        self.py_fns = py_fns
        self.self_name = self_name
        self.batch_size = batch_size
        self.__doc__ = self.py_fns[0].__doc__
        super().__init__(signatures, self_path=self_path, is_method=is_method, is_property=is_property)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        assert len(self.py_fns) > signature_idx
        self.py_fns = [self.py_fns[signature_idx]]

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None

    def _docstring(self) -> Optional[str]:
        return inspect.getdoc(self.py_fns[0])

    @property
    def py_fn(self) -> Callable:
        assert not self.is_polymorphic
        return self.py_fns[0]

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        assert not self.is_polymorphic
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

    def exec_batch(self, args: list[Any], kwargs: dict[str, Any]) -> list:
        """Execute the function with the given arguments and return the result.
        The arguments are expected to be batched: if the corresponding parameter has type T,
        then the argument should have type T if it's a constant parameter, or list[T] if it's
        a batched parameter.
        """
        assert self.is_batched
        assert not self.is_polymorphic
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

    def overload(self, fn: Callable) -> CallableFunction:
        if self.self_path is None:
            raise excs.Error('`overload` can only be used with module UDFs (not locally defined UDFs)')
        if self.is_batched:
            raise excs.Error('`overload` cannot be used with batched functions')
        if self.is_method or self.is_property:
            raise excs.Error('`overload` cannot be used with `is_method` or `is_property`')
        if self._has_resolved_fns:
            raise excs.Error('New `overload` not allowed after the UDF has already been called')
        if self._conditional_return_type is not None:
            raise excs.Error('New `overload` not allowed after a conditional return type has been specified')
        sig = Signature.create(fn)
        self.signatures.append(sig)
        self.py_fns.append(fn)
        return self

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
        assert not self.is_polymorphic  # multi-signature UDFs not allowed for stored fns
        md = {
            'signature': self.signature.as_dict(),
            'batch_size': self.batch_size,
        }
        return md, cloudpickle.dumps(self.py_fn)

    @classmethod
    def from_store(cls, name: Optional[str], md: dict, binary_obj: bytes) -> Function:
        py_fn = cloudpickle.loads(binary_obj)
        assert callable(py_fn)
        sig = Signature.from_dict(md['signature'])
        batch_size = md['batch_size']
        return CallableFunction([sig], [py_fn], self_name=name, batch_size=batch_size)

    def validate_call(self, bound_args: dict[str, Any]) -> None:
        from pixeltable import exprs

        assert not self.is_polymorphic
        if self.is_batched:
            signature = self.signatures[0]
            for param in signature.constant_parameters:
                if param.name in bound_args and isinstance(bound_args[param.name], exprs.Expr):
                    raise ValueError(
                        f'{self.display_name}(): '
                        f'parameter {param.name} must be a constant value, not a Pixeltable expression'
                    )

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'

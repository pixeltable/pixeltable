"""Functions implemented by Python callables (upstream `pixeltable.func.callable_function`)."""

from __future__ import annotations

import base64
import inspect
from typing import Any, Callable, Sequence

import cloudpickle  # type: ignore[import-untyped]

import pixeltable.exceptions as excs

from ._function import Function, InvalidFunction
from ._signature import Signature


class CallableFunction(Function):
    """A Pixeltable function backed by a Python callable.

    Functions defined in modules are serialized by path; others (lambdas, locally defined functions) are pickled
    when serialized and cannot be stored in computed columns.
    """

    py_fns: list[Callable]
    self_name: str | None
    _display_name: str | None
    batch_size: int | None

    def __init__(
        self,
        signatures: list[Signature],
        py_fns: list[Callable],
        self_path: str | None = None,
        self_name: str | None = None,
        display_name: str | None = None,
        batch_size: int | None = None,
        is_method: bool = False,
        is_property: bool = False,
        is_deterministic: bool = True,
    ):
        assert len(signatures) > 0
        assert len(signatures) == len(py_fns)
        if self_path is None and len(signatures) > 1:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                'Multiple signatures are only allowed for module UDFs (not locally defined UDFs)',
            )
        self.py_fns = py_fns
        self.self_name = self_name
        self._display_name = display_name
        self.batch_size = batch_size
        self.__doc__ = self.py_fns[0].__doc__
        super().__init__(
            signatures, self_path=self_path, is_method=is_method, is_property=is_property, is_deterministic=is_deterministic
        )

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        assert len(self.py_fns) > signature_idx
        self.py_fns = [self.py_fns[signature_idx]]

    def _core_spec(self) -> dict[str, Any]:
        return {
            'kind': 'callable',
            'name': self.name,
            'display_name': self.display_name,
            'self_path': self.self_path,
            'signatures': [sig._to_core() for sig in self.signatures],
            'py_fns': self.py_fns,
            'batch_size': self.batch_size,
            'has_resource_pool': self._has_resource_pool,
            'is_method': self.is_method,
            'is_property': self.is_property,
        }

    @property
    def is_batched(self) -> bool:
        return self.batch_size is not None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.py_fns[0])

    def comment(self) -> str | None:
        return inspect.getdoc(self.py_fns[0])

    @property
    def py_fn(self) -> Callable:
        assert not self.is_polymorphic
        return self.py_fns[0]

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        """Calls the function on Python values (not expressions)."""
        assert not self.is_polymorphic
        from ._runtime import run_coroutine

        if self.is_batched:
            constant_param_names = [p.name for p in self.signature.constant_parameters]
            batched_args = [[arg] for arg in args]
            constant_kwargs = {k: v for k, v in kwargs.items() if k in constant_param_names}
            batched_kwargs = {k: [v] for k, v in kwargs.items() if k not in constant_param_names}
            result = self.py_fn(*batched_args, **constant_kwargs, **batched_kwargs)
            if inspect.iscoroutine(result):
                result = run_coroutine(result)
            return result[0]
        result = self.py_fn(*args, **kwargs)
        if inspect.iscoroutine(result):
            result = run_coroutine(result)
        return result

    def get_batch_size(self, *args: Any, **kwargs: Any) -> int | None:
        return self.batch_size

    @property
    def display_name(self) -> str:
        if self._display_name is not None:
            return self._display_name
        if self.self_name is not None:
            return self.self_name
        return super().display_name

    @property
    def name(self) -> str:
        return self.self_name  # type: ignore[return-value]

    def overload(self, fn: Callable) -> CallableFunction:
        if self.self_path is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                '`overload` can only be used with module UDFs (not locally defined UDFs)',
            )
        if self.is_method or self.is_property:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, '`overload` cannot be used with `is_method` or `is_property`'
            )
        if self._has_resolved_fns or self._has_core:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE, 'New `overload` not allowed after the UDF has already been called'
            )
        if self._conditional_return_type is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                'New `overload` not allowed after a conditional return type has been specified',
            )
        sig = Signature.create(fn)
        self.signatures.append(sig)
        self.py_fns.append(fn)
        return self

    def _as_dict(self) -> dict:
        if self.self_path is None:
            # an anonymous function (from Expr.apply() or a locally defined @pxt.udf) is serialized by value
            assert not self.is_method and not self.is_property
            store_md, binary = self.to_store()
            return {'name': self.name, 'store_md': store_md, 'binary': base64.b64encode(binary).decode('ascii')}
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        if 'binary' in d:
            try:
                return cls.from_store(d['name'], d['store_md'], base64.b64decode(d['binary']))
            except Exception:
                error_msg = (
                    f'the locally defined UDF {d["name"]!r} could not be deserialized. This probably means the UDF\n'
                    'was defined in a notebook (or other interactive environment), and the environment has changed\n'
                    'since the UDF was defined.'
                )
                return InvalidFunction(d['name'], d, error_msg)
        return super()._from_dict(d)

    def to_store(self) -> tuple[dict, bytes]:
        assert not self.is_polymorphic
        md = {'signature': self.signature.as_dict(), 'batch_size': self.batch_size}
        return md, cloudpickle.dumps(self.py_fn)

    @classmethod
    def from_store(cls, name: str | None, md: dict, binary_obj: bytes) -> Function:
        py_fn = cloudpickle.loads(binary_obj)
        assert callable(py_fn)
        sig = Signature.from_dict(md['signature'])
        return CallableFunction([sig], [py_fn], self_name=name, batch_size=md['batch_size'])

    def __repr__(self) -> str:
        return f'<Pixeltable UDF {self.name}>'

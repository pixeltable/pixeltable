"""The `Function` base class (upstream `pixeltable.func.function`).

A `Function` holds what introspection of the Python code produced (signatures, callables); its backend half is a
`pixeltable._core.Udf` (`_core_udf()`, created on first use), which binds calls, builds `FunctionCall` expressions
and evaluates them.
"""

from __future__ import annotations

import abc
import importlib
import inspect
import typing
from copy import copy
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import _core

from ._signature import Signature

if TYPE_CHECKING:
    from ._expr_template import ExprTemplateFunction


def resolve_symbol(symbol_path: str) -> object | None:
    """The object a dotted path names (importing its module), or None if no prefix of the path is a module."""
    path_elems = symbol_path.split('.')
    module: ModuleType | None = None
    i = len(path_elems) - 1
    while i > 0 and module is None:
        module_path = '.'.join(path_elems[:i])
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            if e.name != module_path and not module_path.startswith(f'{e.name}.'):
                raise
            i -= 1
    if i == 0:
        return None
    obj: Any = module
    for el in path_elems[i:]:
        obj = getattr(obj, el)
    return obj


def validate_symbol_path(fn_path: str) -> None:
    path_elems = fn_path.split('.')
    fn_name = path_elems[-1]
    if any(el == '<locals>' for el in path_elems):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'{fn_name}(): nested functions are not supported. Move the function to the module level or into a class.',
        )
    if any(not el.isidentifier() for el in path_elems):
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'{fn_name}(): cannot resolve symbol path {fn_path}. '
            f'Move the function to the module level or into a class.',
        )


class Function(abc.ABC):
    """Base class of Pixeltable functions: an object with signatures that builds expressions when called."""

    signatures: list[Signature]
    self_path: str | None
    is_method: bool
    is_property: bool
    is_deterministic: bool
    _conditional_return_type: Callable[..., ts.ColumnType] | None

    def __init__(
        self,
        signatures: list[Signature],
        self_path: str | None = None,
        is_method: bool = False,
        is_property: bool = False,
        is_deterministic: bool = True,
    ):
        assert isinstance(signatures, list)
        self.signatures = signatures
        self.self_path = self_path
        self.is_method = is_method
        self.is_property = is_property
        self.is_deterministic = is_deterministic
        self._conditional_return_type = None
        self.__resolved_fns: list[Function] = []
        self.__core: _core.Udf | None = None
        self._resource_pool: Callable[..., str | None] = lambda: None
        self._has_resource_pool = False
        self._resource_estimator: Callable[..., dict[str, int]] | None = None

    @property
    def is_valid(self) -> bool:
        return len(self.signatures) > 0

    @property
    def name(self) -> str:
        assert self.self_path is not None
        return self.self_path.split('.')[-1]

    @property
    def display_name(self) -> str:
        if self.self_path is None:
            return '<anonymous>'
        ptf_prefix = 'pixeltable.functions.'
        if self.self_path.startswith(ptf_prefix):
            return self.self_path[len(ptf_prefix) :]
        return self.self_path

    @property
    def is_polymorphic(self) -> bool:
        return len(self.signatures) > 1

    @property
    def is_builtin(self) -> bool:
        return self.self_path is not None and self.self_path.startswith('pixeltable.')

    @property
    def is_storable(self) -> bool:
        """True if the serialized form of this function can be stored in a computed column."""
        return self._core_udf().is_storable

    @property
    def signature(self) -> Signature:
        assert not self.is_polymorphic
        return self.signatures[0]

    @property
    def arity(self) -> int:
        assert not self.is_polymorphic
        return len(self.signature.parameters)

    @property
    def is_async(self) -> bool:
        return False

    def comment(self) -> str | None:
        return None

    def help_str(self) -> str:
        docstring = self.comment()
        display = self.display_name + str(self.signatures[0])
        if docstring is None:
            return display
        return f'{display}\n\n{docstring}'

    # ------------------------------------------------------------------------------------------- backend half

    def _core_spec(self) -> dict[str, Any]:
        """The description `pixeltable._core.Udf` is created from."""
        raise NotImplementedError()

    def _make_core(self) -> Any:
        return _core.Udf(self, self._core_spec())

    def _core_udf(self) -> Any:
        if self.__core is None:
            self.__core = self._make_core()
        return self.__core

    @property
    def _has_core(self) -> bool:
        return self.__core is not None

    @property
    def _resolved_fns(self) -> list[Function]:
        """One `Function` per signature (each an overload resolution with just that signature)."""
        if len(self.__resolved_fns) == 0:
            if len(self.signatures) == 1:
                self.__resolved_fns.append(self)
            else:
                for idx in range(len(self.signatures)):
                    resolution = copy(self)
                    resolution.signatures = [self.signatures[idx]]
                    resolution._Function__resolved_fns = [resolution]  # type: ignore[attr-defined]
                    resolution._Function__core = None  # type: ignore[attr-defined]
                    resolution._update_as_overload_resolution(idx)
                    self.__resolved_fns.append(resolution)
        return self.__resolved_fns

    @property
    def _has_resolved_fns(self) -> bool:
        return len(self.__resolved_fns) > 0

    def _resolved_fn(self, idx: int) -> Function:
        return self._resolved_fns[idx]

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        raise NotImplementedError()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._core_udf().call(*args, **kwargs)

    # ------------------------------------------------------------------------------------------- return types

    def conditional_return_type(self, fn: Callable[..., ts.ColumnType]) -> Callable[..., ts.ColumnType]:
        """Instance decorator for specifying a conditional return type for this function"""
        fn_sig = inspect.signature(fn, eval_str=True)
        for param in fn_sig.parameters.values():
            for self_sig in self.signatures:
                if param.name not in self_sig.parameters:
                    raise ValueError(
                        f'`conditional_return_type` has parameter `{param.name}` that is not in a signature'
                    )
        self._conditional_return_type = fn
        return fn

    def _call_conditional_return_type(self, sig_idx: int, bound_args: dict[str, Any]) -> ts.ColumnType | None:
        """Called by the backend while it builds a call: the conditional return type, or None if it does not
        apply (no conditional return type, or one of its arguments is not a constant)."""
        if self._conditional_return_type is None:
            return None
        resolved = self._resolved_fns[sig_idx]
        crt_kwargs = resolved._assemble_callable_args(self._conditional_return_type, bound_args)
        if crt_kwargs is None:
            return None
        return self._conditional_return_type(**crt_kwargs)

    def _call_resource_pool(self, sig_idx: int, bound_args: dict[str, Any]) -> str | None:
        """Called by the backend before it runs calls: the resource pool of a call (None: no pool, or one of the
        arguments the pool depends on is not a constant)."""
        if not self._has_resource_pool:
            return None
        resolved = self._resolved_fns[sig_idx]
        rp_kwargs = resolved._assemble_callable_args(self._resource_pool, bound_args)
        if rp_kwargs is None:
            return None
        return self._resource_pool(**rp_kwargs)

    def _assemble_callable_args(self, callable: Callable, bound_args: dict[str, Any]) -> dict[str, Any] | None:
        """The kwargs for `callable` (which takes a subset of this function's parameters, typed as expressions or
        as Python values), or None if a parameter that expects a constant gets a non-constant argument."""
        callable_signature = inspect.signature(callable, eval_str=True)
        callable_type_hints = typing.get_type_hints(callable)
        callable_args: dict[str, Any] = {}
        sig = self.signatures[0]
        for param in callable_signature.parameters.values():
            if param.name == '_param_types':
                callable_args[param.name] = {p.name: p.col_type for p in sig.parameters.values()}
                continue
            assert param.name in sig.parameters
            arg: Any
            if param.name in bound_args:
                arg = bound_args[param.name]
            elif sig.parameters[param.name].has_default():
                arg = sig.parameters[param.name].default
            else:
                return None
            type_hint = callable_type_hints.get(param.name)
            if typing.get_origin(type_hint) is not None:
                type_hint = typing.get_origin(type_hint)
            if isinstance(type_hint, type) and issubclass(type_hint, _core.Expr):
                callable_args[param.name] = arg
                continue
            is_literal, val = arg._literal_value()
            if not is_literal:
                return None
            callable_args[param.name] = val
        return callable_args

    # ------------------------------------------------------------------------------------------- templates

    def using(self, **kwargs: Any) -> ExprTemplateFunction:
        """A function with some parameters bound to constants (partial binding)."""
        from ._expr_template import ExprTemplate, ExprTemplateFunction

        templates: list[ExprTemplate] = []
        for sig_idx, expr in self._core_udf().using(kwargs):
            sig = self.signatures[sig_idx]
            residual = [p for p in sig.parameters.values() if p.name not in kwargs]
            templates.append(ExprTemplate(expr, Signature(expr.col_type, residual, sig.is_batched)))
        return ExprTemplateFunction(templates)

    # ------------------------------------------------------------------------------------------- decorators

    def to_sql(self, fn: Callable) -> Callable:
        return fn

    def resource_pool(self, fn: Callable[..., str]) -> Callable[..., str]:
        """Instance decorator for specifying the resource pool of this function"""
        self._resource_pool = fn
        self._has_resource_pool = True
        return fn

    def resource_estimator(self, fn: Callable[..., dict[str, int]]) -> Callable[..., dict[str, int]]:
        """Decorator for specifying the resource estimator of this function."""
        self._resource_estimator = fn
        return fn

    # ------------------------------------------------------------------------------------------- serialization

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self.self_path == other.self_path

    def __hash__(self) -> int:
        return hash(self.self_path)

    def as_dict(self) -> dict[str, Any]:
        """A serialized reference to this function (`from_dict()` turns it back into a function)."""
        classpath = f'{self.__class__.__module__}.{self.__class__.__qualname__}'
        return {'_classpath': classpath, **self._as_dict()}

    def _as_dict(self) -> dict:
        assert self.self_path is not None
        return {'path': self.self_path, 'signatures': [sig.as_dict() for sig in self.signatures]}

    @classmethod
    def from_dict(cls, d: dict) -> Function:
        assert '_classpath' in d
        module_path, class_name = d['_classpath'].rsplit('.', 1)
        class_module = importlib.import_module(module_path)
        func_class = getattr(class_module, class_name)
        assert isinstance(func_class, type) and issubclass(func_class, Function)
        return func_class._from_dict(d)

    @classmethod
    def _from_dict(cls, d: dict) -> Function:
        """Default deserialization: the function at the stored path in the current code."""
        path = d.get('path')
        assert path is not None
        try:
            instance = resolve_symbol(path)
            if isinstance(instance, Function):
                return instance
            if instance is None:
                detail = (
                    '(There is no configured project root; did you forget to run `pxt init` in the '
                    "project's top-level directory?)"
                )
                return InvalidFunction(path, d, f'{path!r} names no importable module. {detail}')
            return InvalidFunction(
                path, d, f'the symbol `{path}` is no longer a UDF. (Was the `@pxt.udf` decorator removed?)'
            )
        except (AttributeError, ImportError):
            return InvalidFunction(path, d, f'the symbol `{path}` no longer exists. (Was the UDF moved or renamed?)')

    def __repr__(self) -> str:
        return f'<Pixeltable function {self.display_name}>'


class InvalidFunction(Function):
    """A stored function that cannot be resolved in the current code."""

    fn_dict: dict[str, Any]
    error_msg: str

    def __init__(self, self_path: str, fn_dict: dict[str, Any], error_msg: str):
        super().__init__([], self_path)
        self.fn_dict = fn_dict
        self.error_msg = error_msg

    @property
    def is_storable(self) -> bool:
        return False

    def _as_dict(self) -> dict:
        return self.fn_dict

    def as_dict(self) -> dict[str, Any]:
        return self.fn_dict

    def _core_spec(self) -> dict[str, Any]:
        return {
            'kind': 'invalid',
            'name': self.self_path.split('.')[-1] if self.self_path else None,
            'display_name': self.display_name,
            'self_path': self.self_path,
            'signatures': [],
            'error_msg': self.error_msg,
        }


def _resolve_function(d: dict) -> Function:
    """Called by the backend to resolve a stored function (`Function.as_dict()`)."""
    try:
        return Function.from_dict(d)
    except (ImportError, AttributeError) as exc:
        path = d.get('path') or d.get('name') or '<unknown>'
        return InvalidFunction(str(path), d, f'the function {path!r} could not be loaded: {exc}')

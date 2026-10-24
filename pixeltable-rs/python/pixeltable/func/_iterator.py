"""Iterators (`@pxt.iterator`): functions that expand each input row into zero or more output rows (upstream
`pixeltable.func.iterator`).

Python introspects the decorated generator function or `PxtIterator` subclass (signature, output schema from the
`TypedDict` it yields, `validate()` and `conditional_output_schema()` hooks); the backend half of a call is a
`pixeltable._core.IteratorCall`, which binds and validates the arguments, holds the outputs, serializes the call and
runs the iterator for each base row of a component view (see ARCHITECTURE.md).
"""

from __future__ import annotations

import abc
import collections.abc
import inspect
import typing
from types import MethodType
from typing import Any, Callable, Generic, Iterator, Self, TypeVar

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import _core

from ._function import resolve_symbol
from ._signature import Signature

T = TypeVar('T')

_POS_COLUMN_NAME = 'pos'


class PxtIterator(abc.ABC, Iterator[T], Generic[T]):
    """Base class of class-based iterators (`@pxt.iterator`)."""

    def __iter__(self) -> Self:
        return self

    @abc.abstractmethod
    def __next__(self) -> T: ...

    def close(self) -> None:
        pass

    def seek(self, pos: int, **kwargs: Any) -> None:
        raise NotImplementedError()

    @classmethod
    def validate(cls, literal_args: dict[str, Any]) -> None:
        pass

    @classmethod
    def conditional_output_schema(cls, bound_args: dict[str, Any]) -> dict[str, type] | None:
        return None


class GeneratingFunction:
    """
    A function that evaluates to iterators over its inputs (the Pixeltable equivalent of a table generating function
    in SQL).
    """

    decorated_callable: Callable
    unstored_cols: list[str]
    fqn: str
    signature: Signature
    has_seek: bool
    is_legacy_retrofit: bool

    _default_output_schema: dict[str, ts.ColumnType] | None
    _conditional_output_schema: Callable[[dict[str, Any]], dict[str, type] | None] | None
    _validate: Callable[[dict[str, Any]], Any] | None

    def __init__(self, decorated_callable: Callable, unstored_cols: list[str], fqn: str | None = None) -> None:
        self.decorated_callable = decorated_callable
        self.unstored_cols = [name.lower() for name in unstored_cols]
        if fqn is None:
            self.fqn = f'{decorated_callable.__module__}.{decorated_callable.__qualname__}'
        else:
            self.fqn = fqn
        self._conditional_output_schema = None
        self._validate = None
        self._core = None
        self.signature = Signature.create(decorated_callable, return_type=ts.JsonType())
        self._infer_properties()
        if len(self.unstored_cols) > 0 and not self.has_seek:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Iterator `{self.fqn}` with `unstored_cols` must implement a `seek()` method.',
            )
        self.is_legacy_retrofit = False

    @property
    def name(self) -> str:
        """The unqualified name, which is the last component of fqn."""
        return self.fqn.rsplit('.', 1)[-1]

    def _infer_properties(self) -> None:
        self.py_sig = inspect.signature(self.decorated_callable, eval_str=True)
        output_schema_type: Any
        iter_fn: Callable

        if isinstance(self.decorated_callable, type):
            # a subclass of PxtIterator
            if not issubclass(self.decorated_callable, PxtIterator):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'@pxt.iterator-decorated class `{self.fqn}` must be a subclass of `pixeltable.PxtIterator`.',
                )
            if self.decorated_callable.__next__ is PxtIterator.__next__:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'@pxt.iterator-decorated class `{self.fqn}` must implement a `__next__()` method.',
                )
            self.has_seek = self.decorated_callable.seek is not PxtIterator.seek

            if not isinstance(self.decorated_callable.validate, MethodType):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'`validate()` method of @pxt.iterator `{self.fqn}` must be a @classmethod.',
                )
            if self.decorated_callable.validate.__func__ is not PxtIterator.validate.__func__:  # type: ignore[attr-defined]
                self._validate = self.decorated_callable.validate

            if not isinstance(self.decorated_callable.conditional_output_schema, MethodType):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'`conditional_output_schema()` method of @pxt.iterator `{self.fqn}` must be a @classmethod.',
                )
            if (
                self.decorated_callable.conditional_output_schema.__func__  # type: ignore[attr-defined]
                is not PxtIterator.conditional_output_schema.__func__  # type: ignore[attr-defined]
            ):
                self._conditional_output_schema = self.decorated_callable.conditional_output_schema

            iter_fn = self.decorated_callable.__next__
            return_type = typing.get_type_hints(iter_fn).get('return')
            element_type = typing.get_origin(return_type) or return_type
            if not isinstance(element_type, type) or not issubclass(element_type, dict):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'`__next__()` method of @pxt.iterator-decorated class `{self.fqn}` '
                    'must have return type `dict` or a subclass of `TypedDict`.',
                )
            output_schema_type = element_type

        else:
            # a function that returns an Iterator[T]
            iter_fn = self.decorated_callable
            self.has_seek = False
            return_type = typing.get_type_hints(iter_fn).get('return')
            return_type_args = typing.get_args(return_type)
            element_type = None
            if len(return_type_args) >= 1:
                element_type = typing.get_origin(return_type_args[0]) or return_type_args[0]
            if (
                typing.get_origin(return_type) is not collections.abc.Iterator
                or not isinstance(element_type, type)
                or not issubclass(element_type, dict)
            ):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'@pxt.iterator-decorated function `{self.fqn}()` '
                    'must have return type `Iterator[dict]`, or `Iterator[T]` for a subclass `T` of `TypedDict`.',
                )
            output_schema_type = element_type

        if not hasattr(output_schema_type, '__orig_bases__') or not hasattr(output_schema_type, '__annotations__'):
            # a plain dict: the output schema must come from a conditional_output_schema
            self._default_output_schema = None
            return

        type_hints = typing.get_type_hints(output_schema_type, include_extras=True)
        self._default_output_schema = {}
        for name, type_ in type_hints.items():
            if name.lower() == _POS_COLUMN_NAME:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'{_POS_COLUMN_NAME!r} is reserved and cannot be the name of an iterator output.',
                )
            col_type = ts.ColumnType.from_python_type(type_)
            if col_type is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_TYPE,
                    f'Could not infer Pixeltable type for output field {name!r} '
                    f'(with Python type `{getattr(type_, "__name__", type_)}`).'
                    '\nThis field was mentioned in the return type '
                    f'`{output_schema_type.__module__}.{output_schema_type.__qualname__}` '
                    f'in function `{iter_fn.__module__}.{iter_fn.__qualname__}()`.',
                )
            self._default_output_schema[name] = col_type

    def call_output_schema(self, bound_args: dict[str, Any]) -> dict[str, ts.ColumnType]:
        if self._conditional_output_schema is None:
            if self._default_output_schema is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_CONFIGURATION,
                    f'Iterator `{self.fqn}` must either return a `TypedDict` or define a `conditional_output_schema`.',
                )
            return self._default_output_schema

        output_schema = self._conditional_output_schema(bound_args)
        if output_schema is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'The `conditional_output_schema` for iterator `{self.fqn}` returned None; '
                'it must return a valid output schema dictionary.',
            )
        result: dict[str, ts.ColumnType] = {}
        for name, type_ in output_schema.items():
            result[name] = type_ if isinstance(type_, ts.ColumnType) else ts.ColumnType.normalize_type(type_)
        return result

    # ------------------------------------------------------------------------------------------- backend half

    def _core_spec(self) -> dict[str, Any]:
        return {
            'fqn': self.fqn,
            'signature': self.signature._to_core(),
            'unstored_cols': self.unstored_cols,
            'has_seek': self.has_seek,
            'error_msg': None,
        }

    def _core_iterator(self) -> _core.Iterator:
        if self._core is None:
            self._core = _core.Iterator(self, self._core_spec())
        return self._core

    def __call__(self, *args: Any, **kwargs: Any) -> GeneratingFunctionCall:
        return GeneratingFunctionCall(self, self._core_iterator().call(*args, **kwargs))

    # called by the backend:

    def _literal_args(self, bound_args: dict[str, Any]) -> dict[str, Any]:
        """The constant arguments (plus defaults of unbound parameters), for `validate()`."""
        literal_args: dict[str, Any] = {}
        for k, v in bound_args.items():
            is_literal, val = v._literal_value()
            if is_literal:
                literal_args[k] = val
        for param_name, param in self.py_sig.parameters.items():
            if param_name not in bound_args and param.default is not inspect.Parameter.empty:
                literal_args[param_name] = param.default
        return literal_args

    def _check_call(self, bound_args: dict[str, Any]) -> dict[str, ts.ColumnType]:
        """Runs the custom validation on the constant arguments of a new call and returns its output schema."""
        if self._validate is not None:
            self._validate(self._literal_args(bound_args))
        return self.call_output_schema(bound_args)

    def _start(self, bound_args: dict[str, Any]) -> Iterator[dict]:
        """An iterator over the outputs for one input row (`bound_args`: Python values)."""
        args_with_defaults = bound_args.copy()
        for param_name, param in self.py_sig.parameters.items():
            if param_name not in bound_args and param.default is not inspect.Parameter.empty:
                args_with_defaults[param_name] = param.default
        if self._validate is not None:
            self._validate(args_with_defaults)
        return iter(self.decorated_callable(**bound_args))

    # ------------------------------------------------------------------------------------------- decorators

    def validate(self, fn: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
        if self._validate is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'@pxt.iterator `{self.fqn}` already defines a `validate()` method.',
            )
        self._validate = fn
        return fn

    def conditional_output_schema(
        self, fn: Callable[[dict[str, Any]], dict[str, type] | None]
    ) -> Callable[[dict[str, Any]], dict[str, type] | None]:
        if self._conditional_output_schema is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'@pxt.iterator `{self.fqn}` already defines a `conditional_output_schema()` method.',
            )
        self._conditional_output_schema = fn
        return fn

    # ------------------------------------------------------------------------------------------- serialization

    def as_dict(self) -> dict[str, Any]:
        return {'fqn': self.fqn}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GeneratingFunction:
        fqn = d['fqn']
        try:
            iterator_obj = resolve_symbol(fqn)
            if isinstance(iterator_obj, GeneratingFunction):
                return iterator_obj
            return InvalidGeneratingFunction(
                fqn,
                d,
                f'the symbol `{fqn}` is no longer a Pixeltable iterator. (Was the `@pxt.iterator` decorator removed?)',
            )
        except (AttributeError, ImportError):
            return InvalidGeneratingFunction(
                fqn, d, f'the symbol `{fqn}` no longer exists. (Was the iterator moved or renamed?)'
            )

    def __repr__(self) -> str:
        return f'<Pixeltable iterator {self.fqn}>'


class InvalidGeneratingFunction(GeneratingFunction):
    """A stored iterator that cannot be resolved in the current code."""

    fn_dict: dict[str, Any]
    error_msg: str

    def __init__(self, fqn: str, d: dict[str, Any], error_msg: str) -> None:
        self.fqn = fqn
        self.fn_dict = d
        self.error_msg = error_msg
        self.unstored_cols = []
        self.has_seek = False
        self.is_legacy_retrofit = False
        self._core = None
        self._validate = None
        self._conditional_output_schema = None
        self._default_output_schema = None

    def _error(self) -> excs.Error:
        return excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'The iterator `{self.fqn}` cannot be used, because\n{self.error_msg}'
        )

    def __call__(self, *args: Any, **kwargs: Any) -> GeneratingFunctionCall:
        raise self._error()

    def call_output_schema(self, bound_args: dict[str, Any]) -> dict[str, ts.ColumnType]:
        raise self._error()

    def _core_spec(self) -> dict[str, Any]:
        return {'fqn': self.fqn, 'signature': None, 'unstored_cols': [], 'has_seek': False, 'error_msg': self.error_msg}

    def as_dict(self) -> dict[str, Any]:
        return self.fn_dict


class GeneratingFunctionCall:
    """A call of an iterator: what `create_view(..., iterator=...)` takes. Holds the backend call
    (`pixeltable._core.IteratorCall`)."""

    it: GeneratingFunction

    def __init__(self, it: GeneratingFunction, core: _core.IteratorCall) -> None:
        self.it = it
        self._core = core

    @property
    def outputs(self) -> dict[str, ts.ColumnType]:
        """The output columns (folded name -> type), including `pos`."""
        return dict(self._core.outputs())

    def display_str(self) -> str:
        return self._core.display_str()

    def __repr__(self) -> str:
        return self.display_str()


class BuiltinGeneratingFunction(GeneratingFunction):
    """An iterator implemented in the backend (e.g. `pixeltable.functions.video.frame_iterator`)."""

    def __init__(self, module: str, name: str) -> None:
        self.fqn = f'{module}.{name}'
        self.__module__ = module
        self.__name__ = name
        self.__qualname__ = name
        self._core = None
        self._validate = None
        self._conditional_output_schema = None
        self._default_output_schema = None
        self.is_legacy_retrofit = False
        self.has_seek = True
        core = self._core_iterator()
        self.unstored_cols = list(core.unstored_cols)

    @property
    def decorated_callable(self) -> Any:  # type: ignore[override]
        return self

    def _core_iterator(self) -> _core.Iterator:
        if self._core is None:
            self._core = _core.Iterator._builtin(self, self.fqn)
        return self._core

    def call_output_schema(self, bound_args: dict[str, Any]) -> dict[str, ts.ColumnType]:
        raise NotImplementedError(f'call_output_schema() of the builtin iterator `{self.fqn}`')


_BUILTIN_ITERATORS: set[str] = set(_core.builtin_iterator_names())


def builtin_iterator(module: str, name: str) -> BuiltinGeneratingFunction | None:
    """The backend's iterator `module.name`, if it implements one."""
    if f'{module}.{name}' in _BUILTIN_ITERATORS:
        return BuiltinGeneratingFunction(module, name)
    return None


def _resolve_iterator(d: dict[str, Any]) -> GeneratingFunction:
    """Called by the backend to resolve a stored iterator (`GeneratingFunction.as_dict()`)."""
    return GeneratingFunction.from_dict(d)


def iterator(*args: Any, **kwargs: Any) -> Any:
    """Decorator for iterators: a generator function, or a subclass of `PxtIterator`."""
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return GeneratingFunction(decorated_callable=args[0], unstored_cols=[])
    unstored_cols = kwargs.pop('unstored_cols', [])
    if len(kwargs) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid @iterator decorator kwargs: {", ".join(kwargs.keys())}'
        )
    if len(args) > 0:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, 'Unexpected @iterator decorator arguments.')

    def decorator(decorated_fn: Callable) -> GeneratingFunction:
        return GeneratingFunction(decorated_callable=decorated_fn, unstored_cols=unstored_cols)

    return decorator

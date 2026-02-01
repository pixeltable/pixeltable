import abc
import collections.abc
import importlib
import inspect
import typing
from dataclasses import dataclass
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, TypeVar, overload

from typing_extensions import Self

from pixeltable import exceptions as excs, exprs, type_system as ts

from .signature import Signature

if TYPE_CHECKING:
    from pixeltable.iterators.base import ComponentIterator


# We'd like to say bound=dict, but mypy inexplicably doesn't understand that a TypedDict is a dict (!)
T = TypeVar('T')


@dataclass(frozen=True)
class IteratorOutput:
    orig_name: str
    is_stored: bool
    col_type: ts.ColumnType

    def as_dict(self) -> dict[str, Any]:
        return {'orig_name': self.orig_name, 'is_stored': self.is_stored, 'col_type': self.col_type.as_dict()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'IteratorOutput':
        return cls(orig_name=d['orig_name'], is_stored=d['is_stored'], col_type=ts.ColumnType.from_dict(d['col_type']))


class PxtIterator(abc.ABC, Iterator[T], Generic[T]):
    def __iter__(self) -> Self:
        return self

    @abc.abstractmethod
    def __next__(self) -> T: ...

    def seek(self, pos: int, **kwargs: Any) -> None:
        raise NotImplementedError()

    @classmethod
    def validate(cls, bound_args: dict[str, Any]) -> None:
        pass

    @classmethod
    def conditional_output_schema(cls, bound_args: dict[str, Any]) -> dict[str, type] | None:
        return None


class GeneratingFunction:
    """
    A function that evaluates to iterators over its inputs.

    It is the Pixeltable equivalent of a table generating function in SQL.

    It is the "lift" of a PxtIterator: a PxtIterator represents a one-to-many expansion of its inputs; the
    corresponding GeneratingFunction represents a one-to-many expansion of *columns* of inputs.
    """

    decorated_callable: Callable
    name: str
    signature: Signature
    unstored_cols: list[str]
    has_seek: bool
    is_legacy_retrofit: bool

    _default_output_schema: dict[str, ts.ColumnType] | None
    _conditional_output_schema: Callable[[dict[str, Any]], dict[str, type]] | None
    _validate: Callable[[dict[str, Any]], bool] | None

    def __init__(self, decorated_callable: Callable, unstored_cols: list[str]) -> None:
        self.name = decorated_callable.__name__
        self.decorated_callable = decorated_callable
        self.unstored_cols = unstored_cols
        self._conditional_output_schema = None
        self._validate = None
        self.signature = Signature.create(decorated_callable, return_type=ts.JsonType())

        self._infer_properties()

        if len(self.unstored_cols) > 0 and not self.has_seek:
            raise excs.Error(f'Iterator `{self.fqn}` with `unstored_cols` must implement a `seek()` method.')

        self.is_legacy_retrofit = False

    def _infer_properties(self) -> None:
        self.py_sig = inspect.signature(self.decorated_callable)
        output_schema_type: type[dict]
        iter_fn: Callable

        if isinstance(self.decorated_callable, type):
            # Case 1: decorating a subclass of PxtIterator
            if not issubclass(self.decorated_callable, PxtIterator):
                raise excs.Error(
                    f'@pxt.iterator-decorated class `{self.fqn}` must be a subclass of `pixeltable.PxtIterator`.'
                )
            if self.decorated_callable.__next__ is PxtIterator.__next__:
                raise excs.Error('@pxt.iterator-decorated class `{self.fqn}` must implement a `__next__()` method.')
            self.has_seek = self.decorated_callable.seek is not PxtIterator.seek

            if not isinstance(self.decorated_callable.validate, MethodType):
                raise excs.Error(f'`validate()` method of @pxt.iterator `{self.fqn}` must be a @classmethod.')
            assert isinstance(PxtIterator.validate, MethodType)
            if self.decorated_callable.validate.__func__ is not PxtIterator.validate.__func__:
                # The PxtIterator subclass defines a validate() method; use it as the validator (but strip the `cls`
                # parameter)
                self._validate = self.decorated_callable.validate

            if not isinstance(self.decorated_callable.conditional_output_schema, MethodType):
                raise excs.Error(
                    f'`conditional_output_schema()` method of @pxt.iterator `{self.fqn}` must be a @classmethod.'
                )
            assert isinstance(PxtIterator.conditional_output_schema, MethodType)
            if (
                self.decorated_callable.conditional_output_schema.__func__
                is not PxtIterator.conditional_output_schema.__func__
            ):
                # The PxtIterator subclass defines a conditional_output_schema() method; use it
                self._conditional_output_schema = self.decorated_callable.conditional_output_schema

            iter_fn = self.decorated_callable.__next__
            return_type = typing.get_type_hints(iter_fn).get('return')

            # remove type args from return_type (e.g., convert `dict[str, Any]` to `dict`)
            element_type = typing.get_origin(return_type) or return_type
            if not isinstance(element_type, type) or not issubclass(element_type, dict):
                raise excs.Error(
                    f'`__next__()` method of @pxt.iterator-decorated class `{self.fqn}` '
                    'must have return type `dict` or `MyTypedDict`.'
                )
            output_schema_type = element_type

        else:
            # Case 2: decorating a function that returns an Iterator[T]
            iter_fn = self.decorated_callable
            self.has_seek = False
            return_type = typing.get_type_hints(iter_fn).get('return')

            # Allowed return_types: Iterator[dict], Iterator[dict[str, Any]], Iterator[MyTypedDict]
            return_type_args = typing.get_args(return_type)
            element_type = None
            if len(return_type_args) >= 1:
                # element_type is calculated so that in the above cases it's (respectively):
                # dict, dict, MyTypedDict
                element_type = typing.get_origin(return_type_args[0]) or return_type_args[0]
            if (
                typing.get_origin(return_type) is not collections.abc.Iterator
                or not isinstance(element_type, type)
                or not issubclass(element_type, dict)
            ):
                raise excs.Error(
                    f'@pxt.iterator-decorated function `{self.fqn}()` '
                    'must have return type `Iterator[dict]` or `Iterator[MyTypedDict]`.'
                )
            output_schema_type = element_type

        if not hasattr(output_schema_type, '__orig_bases__') or not hasattr(output_schema_type, '__annotations__'):
            # The return type is a dict, but not a TypedDict. There is no way to infer the output schema at this stage;
            # the user must later define an appropriate conditional_output_schema.
            self._default_output_schema = None
            return

        annotations = output_schema_type.__annotations__.items()
        self._default_output_schema = {}
        for name, type_ in annotations:
            col_type = ts.ColumnType.from_python_type(type_)
            if col_type is None:
                raise excs.Error(
                    f'Could not infer Pixeltable type for output field {name!r} (with Python type `{type_.__name__}`).'
                    '\nThis field was mentioned in the return type '
                    f'`{output_schema_type.__module__}.{output_schema_type.__qualname__}` '
                    f'in function `{iter_fn.__module__}.{iter_fn.__qualname__}()`.'
                )
            self._default_output_schema[name] = col_type

    def call_output_schema(self, bound_kwargs: dict[str, Any]) -> dict[str, ts.ColumnType]:
        if self._conditional_output_schema is None:
            if self._default_output_schema is None:
                raise excs.Error(
                    f'Iterator `{self.fqn}` must either return a `TypedDict` or define a `conditional_output_schema`.'
                )
            return self._default_output_schema

        else:
            output_schema = self._conditional_output_schema(bound_kwargs)
            if output_schema is None:
                raise excs.Error(
                    f'The `conditional_output_schema` for iterator `{self.fqn}` returned None; '
                    'it must return a valid output schema dictionary.'
                )
            return {name: ts.ColumnType.from_python_type(type_) for name, type_ in output_schema.items()}

    def __call__(self, *args: Any, **kwargs: Any) -> 'GeneratingFunctionCall':
        args = [exprs.Expr.from_object(arg) for arg in args]
        kwargs = {k: exprs.Expr.from_object(v) for k, v in kwargs.items()}

        # Promptly validate args and kwargs, as much as possible at this stage.
        try:
            bound_args = self.py_sig.bind(*args, **kwargs).arguments
        except TypeError as exc:
            raise excs.Error(f'Invalid iterator arguments: {exc}') from exc

        self.signature.validate_args(bound_args, context=f'in iterator `{self.fqn}`')

        # Build the dict of literal args for validation and output schema determination
        literal_args = {k: v.val for k, v in bound_args.items() if isinstance(v, exprs.Literal)}

        # Also include in literal_args default values for any unbound args that have them
        for param_name, param in self.py_sig.parameters.items():
            if param_name not in bound_args and param.default is not inspect.Parameter.empty:
                literal_args[param_name] = param.default

        # Run custom iterator validation on whatever args are bound to literals at this stage
        if self._validate is not None:
            self._validate(literal_args)

        output_schema = self.call_output_schema(literal_args)

        outputs = {
            name: IteratorOutput(orig_name=name, is_stored=(name not in self.unstored_cols), col_type=col_type)
            for name, col_type in output_schema.items()
        }

        return GeneratingFunctionCall(self, args, kwargs, bound_args, outputs)

    def eval(self, bound_args: dict[str, Any]) -> Iterator[dict]:
        # Run custom iterator validation on fully bound args
        bound_args_with_defaults = bound_args.copy()
        for param_name, param in self.py_sig.parameters.items():
            if param_name not in bound_args and param.default is not inspect.Parameter.empty:
                bound_args_with_defaults[param_name] = param.default
        if self._validate is not None:
            self._validate(bound_args_with_defaults)
        return self.decorated_callable(**bound_args)

    @classmethod
    def _retrofit(cls, iterator_cls: type['ComponentIterator']) -> 'GeneratingFunction':
        it = GeneratingFunction.__new__(GeneratingFunction)
        it.decorated_callable = iterator_cls
        it.signature = Signature.create(iterator_cls, return_type=ts.JsonType())
        it.py_sig = inspect.signature(iterator_cls)

        def call_output_schema(bound_kwargs: dict[str, Any]) -> dict[str, ts.ColumnType]:
            schema, _ = iterator_cls.output_schema(**bound_kwargs)
            return schema

        it.call_output_schema = call_output_schema  # type: ignore[method-assign]
        it._validate = lambda _: None  # Validation in legacy iterators was done in output_schema()
        it.is_legacy_retrofit = True
        return it

    @property
    def fqn(self) -> str:
        return f'{self.decorated_callable.__module__}.{self.decorated_callable.__qualname__}'

    # validate decorator
    def validate(self, fn: Callable[[dict[str, Any]], bool]) -> Callable[[dict[str, Any]], bool]:
        if self._validate is not None:
            raise excs.Error(f'@pxt.iterator `{self.fqn}` already defines a `validate()` method.')
        self._validate = fn
        return fn

    # conditional_output_schema decorator
    def conditional_output_schema(
        self, fn: Callable[[dict[str, Any]], dict[str, type]]
    ) -> Callable[[dict[str, Any]], dict[str, type]]:
        if self._conditional_output_schema is not None:
            raise excs.Error(f'@pxt.iterator `{self.fqn}` already defines a `conditional_output_schema()` method.')
        self._conditional_output_schema = fn
        return fn

    def as_dict(self) -> dict[str, Any]:
        return {'fqn': self.fqn}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'GeneratingFunction':
        from pixeltable.iterators.base import ComponentIterator

        module_name, class_name = d['fqn'].rsplit('.', 1)
        module = importlib.import_module(module_name)
        iterator_cls = getattr(module, class_name)
        # TODO: Validation
        if isinstance(iterator_cls, GeneratingFunction):
            return iterator_cls
        elif isinstance(iterator_cls, type) and issubclass(iterator_cls, ComponentIterator):
            # Support legacy ComponentIterator pattern for backward compatibility
            return cls._retrofit(iterator_cls)
        else:
            raise AssertionError()  # TODO: Validation


@dataclass(frozen=True)
class GeneratingFunctionCall:
    it: GeneratingFunction
    args: list['exprs.Expr']
    kwargs: dict[str, 'exprs.Expr']
    bound_args: dict[str, 'exprs.Expr']
    outputs: dict[str, IteratorOutput] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            'fn': self.it.as_dict(),
            'args': [arg.as_dict() for arg in self.args],
            'kwargs': {k: v.as_dict() for k, v in self.kwargs.items()},
            'outputs': {name: output_info.as_dict() for name, output_info in self.outputs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'GeneratingFunctionCall':
        it = GeneratingFunction.from_dict(d['fn'])
        args = [exprs.Expr.from_dict(arg_dict) for arg_dict in d['args']]
        kwargs = {k: exprs.Expr.from_dict(v_dict) for k, v_dict in d['kwargs'].items()}

        # Bind args and kwargs against the latest version of the iterator defined in code.
        try:
            bound_args = it.py_sig.bind(*args, **kwargs).arguments
        except TypeError:
            raise AssertionError()  # TODO: Validation

        # Deserialize the output schema and validate against the latest version of the iterator defined in code.
        if d['outputs'] is None:
            # For legacy iterators, `outputs` was not persisted, and there is no practical way to reconstruct it as
            # part of the schema migration. In that case we just query the iterator for it, but we lose the ability to
            # sanity check against any schema evolution of the iterator.
            literal_args = {k: v.val for k, v in bound_args.items() if isinstance(v, exprs.Literal)}
            for param_name, param in it.py_sig.parameters.items():
                if param_name not in bound_args and param.default is not inspect.Parameter.empty:
                    literal_args[param_name] = param.default
            output_schema = it.call_output_schema(literal_args)
            if it.is_legacy_retrofit:
                _, unstored_cols = it.decorated_callable.output_schema(literal_args)  # type: ignore[attr-defined]
            else:
                unstored_cols = it.unstored_cols
            outputs = {
                name: IteratorOutput(orig_name=name, is_stored=(name in unstored_cols), col_type=col_type)
                for name, col_type in output_schema.items()
            }
        else:
            outputs = {
                name: IteratorOutput.from_dict(output_info_dict) for name, output_info_dict in d['outputs'].items()
            }

        return cls(it, args, kwargs, bound_args, outputs)


@overload
def iterator(decorated_fn: Callable) -> GeneratingFunction: ...


@overload
def iterator(*, unstored_cols: list[str] | None = None) -> Callable[[Callable], GeneratingFunction]: ...


def iterator(*args, **kwargs):  # type: ignore[no-untyped-def]
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return GeneratingFunction(decorated_callable=args[0], unstored_cols=[])
    else:
        unstored_cols = kwargs.pop('unstored_cols', None)
        if len(kwargs) > 0:
            raise excs.Error(f'Invalid @iterator decorator kwargs: {", ".join(kwargs.keys())}')
        if len(args) > 0:
            raise excs.Error('Unexpected @iterator decorator arguments.')

        def decorator(decorated_fn: Callable) -> GeneratingFunction:
            return GeneratingFunction(decorated_callable=decorated_fn, unstored_cols=unstored_cols)

        return decorator

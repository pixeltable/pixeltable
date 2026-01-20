import inspect
import itertools
import typing
from collections import abc
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, overload

from pixeltable import exceptions as excs, exprs, type_system as ts
from pixeltable.iterators.base import ComponentIterator

from .signature import Signature


class PxtIterator:
    decorated_callable: Callable
    is_class_based: bool
    init_fn: Callable
    _default_output_schema: dict[str, ts.ColumnType] | None
    signature: Signature

    def __init__(self, decorated_callable: Callable, unstored_cols: list[str]) -> None:
        self.decorated_callable = decorated_callable
        self._default_output_schema = self._infer_output_schema(decorated_callable)
        self.signature = Signature.create(decorated_callable, return_type=ts.JsonType())

    def _infer_output_schema(self, decorated_callable: Callable) -> dict[str, ts.ColumnType] | None:
        if isinstance(decorated_callable, type):
            if not hasattr(decorated_callable, '__init__') or not hasattr(decorated_callable, '__iter__'):
                raise excs.Error(
                    '@pxt.iterator-decorated class '
                    f'`{decorated_callable.__module__}.{decorated_callable.__qualname__}` '
                    'must implement `__init__()` and `__iter__()` methods.'
                )
            self.is_class_based = True
            self.init_fn = decorated_callable.__init__
            iter_fn = decorated_callable.__iter__

        else:
            self.is_class_based = False
            self.init_fn = decorated_callable
            iter_fn = decorated_callable

        py_sig = inspect.signature(iter_fn)
        return_type = py_sig.return_annotation
        return_type_args = typing.get_args(return_type)
        if (
            typing.get_origin(return_type) is not abc.Iterator
            or len(return_type_args) != 1
            or not isinstance(return_type_args[0], type)
            or not issubclass(return_type_args[0], dict)
        ):
            raise excs.Error(
                '@pxt.iterator-decorated function '
                f'`{iter_fn.__module__}.{iter_fn.__qualname__}()` '
                'must have return type `Iterator[dict]` or `Iterator[MyTypedDict]`.'
            )
        output_schema_type = return_type_args[0]

        if not hasattr(output_schema_type, '__orig_bases__') or not hasattr(output_schema_type, '__annotations__'):
            # The return type is a dict, but not a TypedDict. There is no way to infer the output schema at this stage;
            # the user must later define an appropriate conditional_output_schema.
            return None

        annotations = output_schema_type.__annotations__.items()
        output_schema: dict[str, ts.ColumnType] = {}
        for name, type_ in annotations:
            col_type = ts.ColumnType.from_python_type(type_)
            if col_type is None:
                raise excs.Error(
                    f'Could not infer Pixeltable type for output field {name!r} (with Python type `{type_.__name__}`).'
                    '\nThis field was mentioned in the return type '
                    f'`{output_schema_type.__module__}.{output_schema_type.__qualname__}` '
                    f'in iterator function `{iter_fn.__module__}.{iter_fn.__qualname__}()`.'
                )
            output_schema[name] = col_type

        return output_schema

    def output_schema(self, **kwargs: Any) -> dict[str, ts.ColumnType]:
        assert self._default_output_schema is not None
        return self._default_output_schema

    def __call__(self, *args: Any, **kwargs: Any) -> 'IteratorCall':
        py_sig = inspect.signature(self.init_fn)
        args = [exprs.Expr.from_object(arg) for arg in args]
        kwargs = {k: exprs.Expr.from_object(v) for k, v in kwargs.items()}

        if self.is_class_based:
            args = [self.decorated_callable, *args]

        # Prompt validation of supplied args and kwargs.
        try:
            bound_args = py_sig.bind(*args, **kwargs).arguments
        except TypeError as exc:
            raise excs.Error(f'Invalid iterator arguments: {exc}') from exc

        if self.is_class_based:
            self_param_name = next(iter(py_sig.parameters))  # can't guarantee it's actually 'self'
            del bound_args[self_param_name]

        self.signature.validate_args(bound_args, context=f'in iterator `{self.fqn}`')
        literal_args = {k: v.val if isinstance(v, exprs.Literal) else v for k, v in bound_args.items()}
        output_schema = self.output_schema(**literal_args)

        return IteratorCall(self, args, kwargs, bound_args, output_schema)

    def eval(self, bound_args: dict[str, Any]) -> Iterator[dict]:
        return self.decorated_callable(**bound_args)

    def _retrofit(iterator_cls: type[ComponentIterator], iterator_args: dict[str, Any]) -> 'PxtIterator':
        it = PxtIterator.__new__(PxtIterator)
        it.decorated_callable = iterator_cls.__init__
        it._default_output_schema = iterator_cls.output_schema()
        it.signature = Signature.create(iterator_cls.__init__, return_type=ts.JsonType())

    @property
    def fqn(self) -> str:
        return f'{self.decorated_callable.__module__}.{self.decorated_callable.__qualname__}'


@dataclass
class IteratorCall:
    it: PxtIterator
    args: list['exprs.Expr']
    kwargs: dict[str, 'exprs.Expr']
    bound_args: dict[str, 'exprs.Expr']
    output_schema: dict[str, ts.ColumnType]


@overload
def iterator(decorated_fn: Callable) -> PxtIterator: ...


@overload
def iterator(*, unstored_cols: list[str] | None = None) -> Callable[[Callable], PxtIterator]: ...


def iterator(*args, **kwargs):  # type: ignore[no-untyped-def]
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return PxtIterator(decorated_callable=args[0], unstored_cols=[])
    else:
        unstored_cols = kwargs.pop('unstored_cols', None)
        if len(kwargs) > 0:
            raise excs.Error(f'Invalid @iterator decorator kwargs: {", ".join(kwargs.keys())}')
        if len(args) > 0:
            raise excs.Error('Unexpected @iterator decorator arguments.')

        def decorator(decorated_fn: Callable) -> PxtIterator:
            return PxtIterator(decorated_callable=decorated_fn, unstored_cols=unstored_cols)

        return decorator

import inspect
import typing
from collections import abc
from dataclasses import dataclass
from typing import Any, Callable, Iterator, NamedTuple, overload

from pixeltable import exceptions as excs, exprs, type_system as ts
from pixeltable.iterators.base import ComponentIterator

from .signature import Signature


class PxtIterator:
    py_fn: Callable
    _default_output_schema: dict[str, ts.ColumnType] | None
    signature: Signature

    def __init__(self, py_fn: Callable, unstored_cols: list[str]) -> None:
        self.py_fn = py_fn
        self._default_output_schema = self._infer_output_schema(py_fn)
        self.signature = Signature.create(py_fn, return_type=ts.JsonType())

    @classmethod
    def _infer_output_schema(cls, py_fn: Callable) -> dict[str, ts.ColumnType] | None:
        py_sig = inspect.signature(py_fn)
        return_type = py_sig.return_annotation
        return_type_args = typing.get_args(return_type)
        if (
            typing.get_origin(return_type) is not abc.Iterator
            or len(return_type_args) != 1
            or not isinstance(return_type_args[0], type)
            or not issubclass(return_type_args[0], dict)
        ):
            raise excs.Error(
                f'@pxt.iterator-decorated function `{py_fn.__name__}` must have return type Iterator[dict] or Iterator[MyTypedDict]'
            )

        output_schema_type = typing.get_args(return_type)[0]
        if not hasattr(output_schema_type, '__orig_bases__') or not hasattr(output_schema_type, '__annotations__'):
            return None  # Not a TypedDict
        annotations = output_schema_type.__annotations__
        output_schema: dict[str, ts.ColumnType] = {}
        for name, type_ in annotations.items():
            col_type = ts.ColumnType.from_python_type(type_)
            if col_type is None:
                raise excs.Error(
                    f'Could not infer Pixeltable type for output field {name!r} (with Python type `{type_.__name__}`).\n'
                    f'This field was mentioned in the return type `{output_schema_type.__name__}` '
                    f'in iterator function `{py_fn.__name__}`.'
                )
            output_schema[name] = col_type
        return output_schema

    def output_schema(self, **kwargs: Any) -> dict[str, ts.ColumnType]:
        assert self._default_output_schema is not None
        return self._default_output_schema

    def __call__(self, *args: Any, **kwargs: Any) -> 'IteratorCall':
        py_sig = inspect.signature(self.py_fn)
        args = [exprs.Expr.from_object(arg) for arg in args]
        kwargs = {k: exprs.Expr.from_object(v) for k, v in kwargs.items()}

        # Prompt validation of supplied args and kwargs.
        try:
            bound_args = py_sig.bind(*args, **kwargs).arguments
        except TypeError as exc:
            raise excs.Error(f'Invalid iterator arguments: {exc}') from exc

        # self_param_name = next(iter(py_sig.parameters))  # can't guarantee it's actually 'self'
        # del bound_args[self_param_name]

        self.signature.validate_args(bound_args, context=f'in iterator `{self.fqn}`')
        literal_args = {k: v.val if isinstance(v, exprs.Literal) else v for k, v in bound_args.items()}
        output_schema = self.output_schema(**literal_args)

        return IteratorCall(self, args, kwargs, bound_args, output_schema)

    def eval(self, bound_args: dict[str, Any]) -> Iterator[dict]:
        return self.py_fn(**bound_args)

    def _retrofit(iterator_cls: type[ComponentIterator], iterator_args: dict[str, Any]) -> 'PxtIterator':
        it = PxtIterator.__new__(PxtIterator)
        it.py_fn = iterator_cls.__init__
        it._default_output_schema = iterator_cls.output_schema()
        it.signature = Signature.create(iterator_cls.__init__, return_type=ts.JsonType())

    @property
    def fqn(self) -> str:
        return f'{self.py_fn.__module__}.{self.py_fn.__qualname__}'


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
        return PxtIterator(py_fn=args[0], unstored_cols=[])
    else:
        unstored_cols = kwargs.pop('unstored_cols', None)
        if len(kwargs) > 0:
            raise excs.Error(f'Invalid @iterator decorator kwargs: {", ".join(kwargs.keys())}')
        if len(args) > 0:
            raise excs.Error('Unexpected @iterator decorator arguments.')

        def decorator(decorated_fn: Callable) -> PxtIterator:
            return PxtIterator(py_fn=decorated_fn, unstored_cols=unstored_cols)

        return decorator

from collections import abc
import inspect
import typing
from typing import Any, Callable, Iterator, overload
from .signature import Signature
from pixeltable import exceptions as excs, exprs, type_system as ts


class PxtIterator:
    py_fn: Callable
    output_schema: dict[str, ts.ColumnType] | None
    signature: Signature

    def __init__(self, py_fn: Callable, unstored_cols: list[str]) -> None:
        self.py_fn = py_fn
        self.output_schema = self.infer_output_schema(py_fn)
        self.signature = Signature.create(py_fn, return_type=ts.JsonType())

    @classmethod
    def infer_output_schema(cls, py_fn: Callable) -> dict[str, ts.ColumnType] | None:
        py_sig = inspect.signature(py_fn)
        return_type = py_sig.return_annotation
        return_type_args = typing.get_args(return_type)
        if typing.get_origin(return_type) is not abc.Iterator or len(return_type_args) != 1 or not isinstance(return_type_args[0], type) or not issubclass(return_type_args[0], dict):
            raise excs.Error(f'@pxt.iterator-decorated function `{py_fn.__name__}` must have return type Iterator[dict] or Iterator[MyTypedDict]')

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


class IteratorCall:
    it: PxtIterator
    args: list['exprs.Expr']
    kwargs: dict[str, 'exprs.Expr']


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

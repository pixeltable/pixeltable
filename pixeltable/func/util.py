from __future__ import annotations

import inspect
import typing
from typing import List, Callable, Union, Optional, overload

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .batched_function import ExplicitExternalFunction
from .function import Function
from .function_md import FunctionMd
from .globals import resolve_symbol
from .signature import Signature


# Decorator invoked without parentheses: @pxt.udf
@overload
def udf(fn: Callable) -> Function: ...


# Decorator schema invoked with parentheses: @pxt.udf(**kwargs)
@overload
def udf(
        *,
        param_types: Optional[List[ts.ColumnType]] = None,
        return_type: Optional[ts.ColumnType] = None,
        batch_size: Optional[int] = None
) -> Callable: ...


def udf(*args, **kwargs):
    """A decorator to create a Function from a function definition.

    Examples:
        >>> @pxt.udf
        ... def my_function(x: int) -> int:
        ...    return x + 1

        >>> @pxt.udf(param_types=[pxt.IntType()], return_type=pxt.IntType())
        ... def my_function(x):
        ...    return x + 1
    """

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):

        # Decorator invoked without parentheses: @pxt.udf
        # Simply call make_function with defaults.
        return make_function(None, None, args[0])

    else:

        # Decorator invoked with parentheses: @pxt.udf(**kwargs)
        return_type = kwargs.pop('return_type', None)
        param_types = kwargs.pop('param_types', None)
        batch_size = kwargs.pop('batch_size', None)

        def decorator(fn: Callable):
            return make_function(return_type, param_types, fn, batch_size=batch_size)
        return decorator


T = typing.TypeVar('T')
Batch = typing.Annotated[list[T], 'pxt-batch']


def _unpack_batch_type(t: type) -> Optional[type]:
    if typing.get_origin(t) == typing.Annotated:
        batch_args = typing.get_args(t)
        if len(batch_args) == 2 and batch_args[1] == 'pxt-batch':
            assert typing.get_origin(batch_args[0]) == list
            list_args = typing.get_args(batch_args[0])
            return list_args[0]
    return None


def make_function(
    return_type: Optional[ts.ColumnType],
    param_types: Optional[List[ts.ColumnType]],
    eval_fn: Callable,
    display_name: Optional[str] = None,
    batch_size: Optional[int] = None
) -> Function:
    assert eval_fn is not None
    if return_type is None:
        if 'return' in typing.get_type_hints(eval_fn):
            py_return_type = typing.get_type_hints(eval_fn, include_extras=True)['return']
            if py_return_type is not None:
                if batch_size is None:
                    return_type = ts.ColumnType.from_python_type(py_return_type)
                else:
                    # batch_size specified
                    batch_type = _unpack_batch_type(py_return_type)
                    if batch_type is None:
                        raise excs.Error(f'`batch_size is specified; Python return type must be a `Batch`')
                    return_type = ts.ColumnType.from_python_type(batch_type)
        if return_type is None:
            raise excs.Error(f'Cannot infer pixeltable result type. Specify `return_type` explicitly?')
    constant_params = []
    if param_types is None:
        py_signature = inspect.signature(eval_fn)
        param_types = []
        for param_name, py_type in typing.get_type_hints(eval_fn, include_extras=True).items():
            if param_name != 'return':
                batch_type = _unpack_batch_type(py_type)
                if batch_size is not None and batch_type is not None:
                    col_type = ts.ColumnType.from_python_type(batch_type)
                    if col_type is None:
                        raise excs.Error(f'Cannot infer pixeltable type of parameter: `{param_name}`. Specify `param_types` explicitly?')
                else:
                    constant_params.append(param_name)
                    col_type = ts.ColumnType.from_python_type(py_type)
                    if col_type is None:
                        raise excs.Error(f'Cannot infer pixeltable type of parameter: `{param_name}`. Specify `param_types` explicitly?')
                param_types.append(col_type)
        if len(param_types) != len(py_signature.parameters):
            raise excs.Error(f'Cannot infer pixeltable types of parameters. Specify `param_types` explicitly?')
    signature = Signature.create(eval_fn, False, param_types, return_type)
    md = FunctionMd(signature, False, False)
    try:
        md.src = inspect.getsource(eval_fn)
    except OSError as e:
        pass
    if batch_size is None:
        return Function(md, eval_fn=eval_fn, display_name=display_name)
    else:
        # batch_size is specified
        return ExplicitExternalFunction(md, batch_size=batch_size, invoker_fn=eval_fn, constant_params=constant_params, display_name=display_name)


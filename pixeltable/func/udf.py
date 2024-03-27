from __future__ import annotations

import inspect
import typing
from typing import List, Callable, Optional, overload

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from . import Signature, Function, CallableFunction, ExplicitBatchedFunction, FunctionRegistry


# Decorator invoked without parentheses: @pxt.udf
@overload
def udf(fn: Callable) -> Function: ...


# Decorator schema invoked with parentheses: @pxt.udf(**kwargs)
@overload
def udf(
        *,
        return_type: Optional[ts.ColumnType] = None,
        param_types: Optional[List[ts.ColumnType]] = None,
        batch_size: Optional[int] = None,
        py_fn: Optional[Callable] = None
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
        return make_function(eval_fn=args[0])

    else:

        # Decorator schema invoked with parentheses: @pxt.udf(**kwargs)
        # Create a decorator for the specified schema.
        return_type = kwargs.pop('return_type', None)
        param_types = kwargs.pop('param_types', None)
        batch_size = kwargs.pop('batch_size', None)
        py_fn = kwargs.pop('py_fn', None)

        def decorator(fn: Callable):
            return make_function(fn, return_type, param_types, batch_size, substitute_fn=py_fn)

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
    eval_fn: Callable,
    return_type: Optional[ts.ColumnType] = None,
    param_types: Optional[List[ts.ColumnType]] = None,
    batch_size: Optional[int] = None,
    substitute_fn: Optional[Callable] = None,
    function_name: Optional[str] = None
) -> Function:
    """
    Constructs a `CallableFunction` or `BatchedFunction`, depending on the
    supplied parameters. If `substitute_fn` is specified, then `eval_fn`
    will be used only for its signature, with execution delegated to
    `substitute_fn`.
    """
    # Obtain function_path from eval_fn when appropriate
    if eval_fn.__module__ != '__main__' and eval_fn.__name__.isidentifier():
        function_path = f'{eval_fn.__module__}.{eval_fn.__qualname__}'
    else:
        function_path = None

    # Derive function_name, if not specified explicitly
    if function_name is None:
        function_name = eval_fn.__name__

    # Display name to use for error messages
    errmsg_name = function_name if function_path is None else function_path

    # Attempt to infer `return_type`, if not specified explicitly;
    # validate that batched functions must have a batched return type.

    if return_type is None and 'return' in typing.get_type_hints(eval_fn):
        py_return_type = typing.get_type_hints(eval_fn, include_extras=True)['return']
        if py_return_type is not None:
            if batch_size is None:
                return_type = ts.ColumnType.from_python_type(py_return_type)
            else:
                # batch_size specified
                batch_type = _unpack_batch_type(py_return_type)
                if batch_type is None:
                    raise excs.Error(f'{errmsg_name}(): batch_size is specified; Python return type must be a `Batch`')
                return_type = ts.ColumnType.from_python_type(batch_type)

    if return_type is None:
        raise excs.Error(f'{errmsg_name}(): Cannot infer pixeltable result type. Specify `return_type` explicitly?')

    py_signature = inspect.signature(eval_fn)

    # Attempt to infer parameter types, if not specified explicitly;
    # validate batched parameters; and identify `constant_params`.

    if param_types is None:
        infer_param_types = True
        param_types = []
    else:
        infer_param_types = False
    constant_params = []

    for param_name, py_type in typing.get_type_hints(eval_fn, include_extras=True).items():
        if param_name != 'return':

            batch_type = _unpack_batch_type(py_type)
            if batch_type is not None:
                if batch_size is None:
                    raise excs.Error(
                        f'{errmsg_name}(): Batched parameter in udf, but no `batch_size` given: `{param_name}`'
                    )
                unpacked_type = batch_type
            else:
                if batch_size is not None:
                    constant_params.append(param_name)
                unpacked_type = py_type

            if infer_param_types:
                col_type = ts.ColumnType.from_python_type(unpacked_type)
                if col_type is None:
                    raise excs.Error(
                        f'{errmsg_name}(): Cannot infer pixeltable type of parameter: `{param_name}`. '
                        'Specify `param_types` explicitly?'
                    )
                param_types.append(col_type)

    if infer_param_types and len(param_types) != len(py_signature.parameters):
        raise excs.Error(f'{errmsg_name}(): Cannot infer pixeltable types of parameters. Specify `param_types` explicitly?')

    if substitute_fn is None:
        py_fn = eval_fn
    else:
        if function_path is None:
            raise excs.Error(f'{errmsg_name}(): The @udf decorator with the explicit py_fn argument can only be used in a module')
        py_fn = substitute_fn

    signature = Signature.create(py_fn, param_types, return_type)

    if batch_size is None:
        result = CallableFunction(signature=signature, py_fn=py_fn, self_path=function_path, self_name=function_name)
    else:
        result = ExplicitBatchedFunction(
            signature=signature, batch_size=batch_size, invoker_fn=eval_fn,
            constant_params=constant_params, self_path=function_path)

    # If this function is part of a module, register it
    if function_path is not None:
        FunctionRegistry.get().register_function(function_path, result)

    return result

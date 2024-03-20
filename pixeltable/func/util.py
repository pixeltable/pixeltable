from __future__ import annotations

import inspect
import typing
from typing import List, Callable, Union, Optional, Iterable, overload

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from . import ExternalFunction
from .external_function import ExplicitExternalFunction
from .function import Function
from .function_md import FunctionMd
from .globals import resolve_symbol
from .signature import Signature


# Decorator invoked without parentheses: @pxt.udf
@overload
def udf(fn: Callable) -> Function: ...


# Decorator invoked with parentheses: @pxt.udf(**kwargs)
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
            py_return_type = typing.get_type_hints(eval_fn)['return']
            if py_return_type is not None:
                if batch_size is None:
                    return_type = ts.ColumnType.from_python_type(py_return_type)
                else:
                    # batch_size specified
                    if not isinstance(py_return_type, Iterable):
                        raise excs.Error(f'`batch_size is specified; Python return type must be an `Iterable`')
                    return_type = ts.ColumnType.from_python_type(typing.get_args(py_return_type)[0])
        if return_type is None:
            raise excs.Error(f'Cannot infer pixeltable result type. Specify `return_type` explicitly?')
    constant_params = []
    if param_types is None:
        py_signature = inspect.signature(eval_fn)
        param_types = []
        for param_name, py_type in typing.get_type_hints(eval_fn).items():
            if param_name != 'return':
                if batch_size is not None and isinstance(py_type, Iterable):
                    col_type = ts.ColumnType.from_python_type(typing.get_args(py_type)[0])
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


def make_aggregate_function(
        return_type: ts.ColumnType, param_types: List[ts.ColumnType],
        init_fn: Callable, update_fn: Callable, value_fn: Callable,
        requires_order_by: bool = False, allows_std_agg: bool = False, allows_window: bool = False
) -> Function:
    assert init_fn is not None and update_fn is not None and value_fn is not None
    signature = Signature.create(update_fn, True, param_types, return_type)
    md = FunctionMd(signature, True, False)
    md.requires_order_by = requires_order_by
    md.allows_std_agg = allows_std_agg
    md.allows_window = allows_window
    try:
        md.src = (
            f'init:\n{inspect.getsource(init_fn)}\n\n'
            f'update:\n{inspect.getsource(update_fn)}\n\n'
            f'value:\n{inspect.getsource(value_fn)}\n'
        )
    except OSError as e:
        pass
    return Function(md, init_fn=init_fn, update_fn=update_fn, value_fn=value_fn)

def make_library_function(
        return_type: Union[ts.ColumnType, Callable], param_types: List[ts.ColumnType], module_name: str,
        eval_symbol: str
) -> Function:
    assert module_name is not None and eval_symbol is not None
    eval_fn = resolve_symbol(module_name, eval_symbol)
    signature = Signature.create(eval_fn, False, param_types, return_type)
    md = FunctionMd(signature, False, True)
    return Function(md, module_name=module_name, eval_symbol=eval_symbol)

def make_library_aggregate_function(
        return_type: ts.ColumnType, param_types: List[ts.ColumnType],
        module_name: str, init_symbol: str, update_symbol: str, value_symbol: str,
        requires_order_by: bool = False, allows_std_agg: bool = False, allows_window: bool = False
) -> Function:
    assert module_name is not None and init_symbol is not None and update_symbol is not None \
           and value_symbol is not None
    update_fn = resolve_symbol(module_name, update_symbol)
    signature = Signature.create(update_fn, True, param_types, return_type)
    md = FunctionMd(signature, True, True)
    md.requires_order_by = requires_order_by
    md.allows_std_agg = allows_std_agg
    md.allows_window = allows_window
    return Function(
        md, module_name=module_name, init_symbol=init_symbol, update_symbol=update_symbol,
        value_symbol=value_symbol)

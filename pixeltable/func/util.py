from __future__ import annotations
from typing import List, Callable, Union
import inspect


from .globals import resolve_symbol
from .signature import Signature, Parameter
from .function_md import FunctionMd
from .function import Function
import pixeltable.type_system as ts


def udf(*, return_type: ts.ColumnType, param_types: List[ts.ColumnType]) -> Callable:
    """Returns decorator to create a Function from a function definition.

    Example:
        >>> @pt.udf(param_types=[pt.IntType()], return_type=pt.IntType())
        ... def my_function(x):
        ...    return x + 1
    """
    from .function import Function
    def decorator(fn: Callable) -> Function:
        return make_function(return_type, param_types, fn)
    return decorator

def make_function(return_type: ts.ColumnType, param_types: List[ts.ColumnType], eval_fn: Callable) -> Function:
    assert eval_fn is not None
    signature = Signature.create(eval_fn, False, param_types, return_type)
    md = FunctionMd(signature, False, False)
    try:
        md.src = inspect.getsource(eval_fn)
    except OSError as e:
        pass
    return Function(md, eval_fn=eval_fn)

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

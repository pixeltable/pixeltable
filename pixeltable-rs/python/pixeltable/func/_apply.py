"""`Expr.apply(fn)`: a unary function around an arbitrary Python callable (upstream
`Expr._make_applicator_function()`)."""

from __future__ import annotations

import inspect
import json
import typing
from typing import Any, Callable

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from ._callable import CallableFunction
from ._udf import make_function

# result types of common stdlib callables without type hints
_KNOWN_APPLICATOR_TYPES: dict[Callable, ts.ColumnType] = {
    str: ts.StringType(),
    json.dumps: ts.StringType(),
    json.loads: ts.JsonType(),
}


def make_applicator(fn: Callable, col_type: Any, arg_type: ts.ColumnType) -> CallableFunction:
    """Called by `Expr.apply()`: the function that applies `fn` to a value of type `arg_type`."""
    fn_type: ts.ColumnType | None
    if col_type is not None:
        fn_type = ts.ColumnType.normalize_type(col_type)
    elif fn in _KNOWN_APPLICATOR_TYPES:
        fn_type = _KNOWN_APPLICATOR_TYPES[fn]
    elif 'return' in typing.get_type_hints(fn):
        fn_type = ts.ColumnType.from_python_type(typing.get_type_hints(fn)['return'])
    else:
        fn_type = None
    if fn_type is None:
        raise excs.RequestError(
            excs.ErrorCode.UNSUPPORTED_OPERATION,
            f'Column type of `{fn.__name__}` cannot be inferred. Use `.apply({fn.__name__}, col_type=...)` to specify.',
        )

    try:
        params = inspect.signature(fn, eval_str=True).parameters
        params_iter = iter(params.values())
        first_param = next(params_iter) if len(params) >= 1 else None
        second_param = next(params_iter) if len(params) >= 2 else None
        if first_param is None or first_param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Function `{fn.__name__}` has no positional parameters.'
            )
        if (
            second_param is not None
            and second_param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and second_param.default is inspect.Parameter.empty
        ):
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'Function `{fn.__name__}` has multiple required parameters.'
            )
    except ValueError:
        # builtins have no inspectable signature
        pass

    # a unary lambda around `fn` (which may have optional parameters), named after `fn`
    return make_function(
        decorated_fn=lambda x: fn(x), return_type=fn_type, param_types=[arg_type], function_name=fn.__name__
    )

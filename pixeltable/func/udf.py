from __future__ import annotations

import inspect
from typing import List, Callable, Optional, overload, Any

import pixeltable as pxt
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .batched_function import ExplicitBatchedFunction
from .callable_function import CallableFunction
from .expr_template_function import ExprTemplateFunction
from .function import Function
from .function_registry import FunctionRegistry
from .globals import validate_symbol_path
from .signature import Signature


# Decorator invoked without parentheses: @pxt.udf
@overload
def udf(decorated_fn: Callable) -> Function: ...


# Decorator schema invoked with parentheses: @pxt.udf(**kwargs)
@overload
def udf(
        *,
        return_type: Optional[ts.ColumnType] = None,
        param_types: Optional[List[ts.ColumnType]] = None,
        batch_size: Optional[int] = None,
        substitute_fn: Optional[Callable] = None,
        _force_stored: bool = False
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
        return make_function(decorated_fn=args[0])

    else:

        # Decorator schema invoked with parentheses: @pxt.udf(**kwargs)
        # Create a decorator for the specified schema.
        return_type = kwargs.pop('return_type', None)
        param_types = kwargs.pop('param_types', None)
        batch_size = kwargs.pop('batch_size', None)
        substitute_fn = kwargs.pop('py_fn', None)
        force_stored = kwargs.pop('_force_stored', False)

        def decorator(decorated_fn: Callable):
            return make_function(
                decorated_fn, return_type, param_types, batch_size, substitute_fn=substitute_fn,
                force_stored=force_stored)

        return decorator


def make_function(
    decorated_fn: Callable,
    return_type: Optional[ts.ColumnType] = None,
    param_types: Optional[List[ts.ColumnType]] = None,
    batch_size: Optional[int] = None,
    substitute_fn: Optional[Callable] = None,
    function_name: Optional[str] = None,
    force_stored: bool = False
) -> Function:
    """
    Constructs a `CallableFunction` or `BatchedFunction`, depending on the
    supplied parameters. If `substitute_fn` is specified, then `decorated_fn`
    will be used only for its signature, with execution delegated to
    `substitute_fn`.
    """
    # Obtain function_path from decorated_fn when appropriate
    if force_stored:
        # force storing the function in the db
        function_path = None
    elif decorated_fn.__module__ != '__main__' and decorated_fn.__name__.isidentifier():
        function_path = f'{decorated_fn.__module__}.{decorated_fn.__qualname__}'
    else:
        function_path = None

    # Derive function_name, if not specified explicitly
    if function_name is None:
        function_name = decorated_fn.__name__

    # Display name to use for error messages
    errmsg_name = function_name if function_path is None else function_path

    sig = Signature.create(decorated_fn, param_types, return_type)

    # batched functions must have a batched return type
    # TODO: remove 'Python' from the error messages when we have full inference with Annotated types
    if batch_size is not None and not sig.is_batched:
        raise excs.Error(f'{errmsg_name}(): batch_size is specified; Python return type must be a `Batch`')
    if batch_size is not None and len(sig.batched_parameters) == 0:
        raise excs.Error(f'{errmsg_name}(): batch_size is specified; at least one Python parameter must be `Batch`')
    if batch_size is None and len(sig.batched_parameters) > 0:
        raise excs.Error(f'{errmsg_name}(): batched parameters in udf, but no `batch_size` given')

    if substitute_fn is None:
        py_fn = decorated_fn
    else:
        if function_path is None:
            raise excs.Error(f'{errmsg_name}(): @udf decorator with a `substitute_fn` can only be used in a module')
        py_fn = substitute_fn

    if batch_size is None:
        result = CallableFunction(signature=sig, py_fn=py_fn, self_path=function_path, self_name=function_name)
    else:
        result = ExplicitBatchedFunction(
            signature=sig, batch_size=batch_size, invoker_fn=py_fn, self_path=function_path)

    # If this function is part of a module, register it
    if function_path is not None:
        # do the validation at the very end, so it's easier to write tests for other failure scenarios
        validate_symbol_path(function_path)
        FunctionRegistry.get().register_function(function_path, result)

    return result

@overload
def expr_udf(py_fn: Callable) -> ExprTemplateFunction: ...

@overload
def expr_udf(*, param_types: Optional[List[ts.ColumnType]] = None) -> Callable: ...

def expr_udf(*args: Any, **kwargs: Any) -> Any:
    def decorator(py_fn: Callable, param_types: Optional[List[ts.ColumnType]]) -> ExprTemplateFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            # this is a named function in a module
            function_path = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None

        # TODO: verify that the inferred return type matches that of the template
        # TODO: verify that the signature doesn't contain batched parameters

        # construct Parameters from the function signature
        params = Signature.create_parameters(py_fn, param_types=param_types)
        import pixeltable.exprs as exprs
        var_exprs = [exprs.Variable(param.name, param.col_type) for param in params]
        # call the function with the parameter expressions to construct an Expr with parameters
        template = py_fn(*var_exprs)
        assert isinstance(template, exprs.Expr)
        py_sig = inspect.signature(py_fn)
        if function_path is not None:
            validate_symbol_path(function_path)
        return ExprTemplateFunction(template, py_signature=py_sig, self_path=function_path, name=py_fn.__name__)

    if len(args) == 1:
        assert len(kwargs) == 0 and callable(args[0])
        return decorator(args[0], None)
    else:
        assert len(args) == 0 and len(kwargs) == 1 and 'param_types' in kwargs
        return lambda py_fn: decorator(py_fn, kwargs['param_types'])

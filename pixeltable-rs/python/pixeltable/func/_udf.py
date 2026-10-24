"""The `@pxt.udf` and `@pxt.expr_udf` decorators (upstream `pixeltable.func.udf`)."""

from __future__ import annotations

import builtins
import sys
from typing import Any, Callable, Sequence

import __main__
import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import _core

from ._callable import CallableFunction
from ._expr_template import ExprTemplate, ExprTemplateFunction
from ._function import Function, validate_symbol_path
from ._signature import Signature


def is_interactive_env() -> bool:
    """True if running in an interactive environment (IPython/Jupyter or the Python shell)."""
    if getattr(builtins, '__IPYTHON__', False):
        return True
    if hasattr(sys, 'ps1'):
        return True
    return not hasattr(__main__, '__file__') and bool(sys.flags.interactive)


class FunctionRegistry:
    """The registry of module-level functions and of the methods/properties registered per type (held by the
    backend)."""

    _instance: FunctionRegistry | None = None

    @classmethod
    def get(cls) -> FunctionRegistry:
        if cls._instance is None:
            cls._instance = FunctionRegistry()
        return cls._instance

    def register_function(self, fqn: str, fn: Function) -> None:
        method_of: int | None = None
        if fn.is_method or fn.is_property:
            method_of = fn.signatures[0].parameters_by_pos[0].col_type.type_enum.value  # type: ignore[union-attr]
        _core.register_function(fqn, fn.name, fn, method_of)

    def list_functions(self) -> list[Function]:
        return list(_core.registered_functions())


def udf(*args, **kwargs):  # type: ignore[no-untyped-def]
    """A decorator to create a Function from a function definition.

    Examples:
        >>> @pxt.udf
        ... def my_function(x: int) -> int:
        ...     return x + 1
    """
    from pixeltable._table import Table

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]) and not isinstance(args[0], Table):
        return make_function(decorated_fn=args[0], from_decorator=True)

    if len(args) == 1 and isinstance(args[0], Table):
        return_value = kwargs.pop('return_value', None)
        description = kwargs.pop('description', None)
        if len(kwargs) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid udf kwargs: {", ".join(kwargs.keys())}'
            )
        from ._table_udf import from_table

        return from_table(args[0], return_value, description)

    batch_size = kwargs.pop('batch_size', None)
    substitute_fn = kwargs.pop('substitute_fn', None)
    is_method = kwargs.pop('is_method', None)
    is_property = kwargs.pop('is_property', None)
    is_deterministic = kwargs.pop('is_deterministic', None)
    resource_pool = kwargs.pop('resource_pool', None)
    type_substitutions = kwargs.pop('type_substitutions', None)
    display_name = kwargs.pop('display_name', None)
    force_stored = kwargs.pop('_force_stored', False)
    if len(kwargs) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid @udf decorator kwargs: {", ".join(kwargs.keys())}'
        )
    if len(args) > 0:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, 'Unexpected @udf decorator arguments.')

    def decorator(decorated_fn: Callable) -> CallableFunction:
        return make_function(
            decorated_fn,
            batch_size=batch_size,
            substitute_fn=substitute_fn,
            is_method=is_method,
            is_property=is_property,
            is_deterministic=is_deterministic,
            resource_pool=resource_pool,
            type_substitutions=type_substitutions,
            display_name=display_name,
            force_stored=force_stored,
            from_decorator=True,
        )

    return decorator


def make_function(
    decorated_fn: Callable,
    return_type: ts.ColumnType | None = None,
    param_types: list[ts.ColumnType] | None = None,
    batch_size: int | None = None,
    substitute_fn: Callable | None = None,
    is_method: bool | None = False,
    is_property: bool | None = False,
    is_deterministic: bool | None = True,
    resource_pool: str | None = None,
    type_substitutions: Sequence[dict] | None = None,
    function_name: str | None = None,
    display_name: str | None = None,
    force_stored: bool = False,
    from_decorator: bool = False,
) -> CallableFunction:
    """Constructs a `CallableFunction`; with `substitute_fn`, `decorated_fn` supplies only the signature."""
    defined_in_script = False
    function_path: str | None
    if force_stored:
        function_path = None
    elif decorated_fn.__module__ != '__main__' and decorated_fn.__name__.isidentifier():
        function_path = f'{decorated_fn.__module__}.{decorated_fn.__qualname__}'
    elif from_decorator and is_interactive_env() and decorated_fn.__name__.isidentifier():
        # a udf inlined in a notebook cell gets a __main__ path, which resolves only within the session
        function_path = f'{decorated_fn.__module__}.{decorated_fn.__qualname__}'
    else:
        if from_decorator and not hasattr(sys, 'ps1'):
            defined_in_script = True
        function_path = None

    if function_name is None:
        function_name = decorated_fn.__name__
    errmsg_name = function_name if function_path is None else function_path

    if defined_in_script:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION,
            f'Defining the UDF {errmsg_name!r} directly in the global namespace of a Python script\n'
            f'is not allowed. To fix this error, move the UDF {errmsg_name!r} into a named module\n'
            '(a .py file other than your script), then import that module into your script.',
        )

    signatures: list[Signature]
    if type_substitutions is None:
        sig = Signature.create(decorated_fn, param_types, return_type)
        if batch_size is not None and not sig.is_batched:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): batch_size is specified; Python return type must be a `Batch`',
            )
        if batch_size is not None and len(sig.batched_parameters) == 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): batch_size is specified; at least one Python parameter must be `Batch`',
            )
        if batch_size is None and len(sig.batched_parameters) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): batched parameters in udf, but no `batch_size` given',
            )
        if is_method and is_property:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Cannot specify both `is_method` and `is_property` (in function `{function_name}`)',
            )
        if is_property and len(sig.parameters) != 1:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                '`is_property=True` expects a UDF with exactly 1 parameter, but '
                f'`{function_name}` has {len(sig.parameters)}',
            )
        if (is_method or is_property) and function_path is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                'Stored functions cannot be declared using `is_method` or `is_property`',
            )
        signatures = [sig]
    else:
        if function_path is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): type substitutions can only be used with module UDFs (not locally defined UDFs)',
            )
        if batch_size is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): type substitutions cannot be used with batched functions',
            )
        if is_method is not None or is_property is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): type substitutions cannot be used with `is_method` or `is_property`',
            )
        signatures = [
            Signature.create(decorated_fn, param_types, return_type, type_substitutions=subst)
            for subst in type_substitutions
        ]

    if substitute_fn is None:
        py_fn = decorated_fn
    else:
        if function_path is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'{errmsg_name}(): @udf decorator with a `substitute_fn` can only be used in a module',
            )
        py_fn = substitute_fn

    result = CallableFunction(
        signatures=signatures,
        py_fns=[py_fn] * len(signatures),
        self_path=function_path,
        self_name=function_name,
        display_name=display_name,
        batch_size=batch_size,
        is_method=bool(is_method),
        is_property=bool(is_property),
        is_deterministic=True if is_deterministic is None else is_deterministic,
    )
    if resource_pool is not None:
        result.resource_pool(lambda: resource_pool)

    if function_path is not None:
        validate_symbol_path(function_path)
        # a __main__ path (a udf inlined in a notebook cell) is not importable across sessions: not registered
        if decorated_fn.__module__ != '__main__':
            FunctionRegistry.get().register_function(function_path, result)

    return result


def expr_udf(*args: Any, **kwargs: Any) -> Any:
    """A decorator for functions defined by an expression over their parameters."""

    def make_expr_template(
        py_fn: Callable, param_types: list[ts.ColumnType] | None, is_method: bool, is_property: bool
    ) -> ExprTemplateFunction:
        if py_fn.__module__ != '__main__' and py_fn.__name__.isidentifier():
            function_path: str | None = f'{py_fn.__module__}.{py_fn.__qualname__}'
        else:
            function_path = None

        function_name = py_fn.__name__
        if is_method and is_property:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'Cannot specify both `is_method` and `is_property` (in function `{function_name}`)',
            )
        if (is_method or is_property) and function_path is None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'`{function_name}`: expr_udfs declared with `is_method` or `is_property` must live in a module',
            )

        sig = Signature.create(py_fn=py_fn, param_types=param_types, return_type=ts.InvalidType())
        var_exprs = [_core.variable(param.name, param.col_type) for param in sig.parameters.values()]
        # calling the function with the parameter variables yields the template expression
        expr = py_fn(*var_exprs)
        if not isinstance(expr, _core.Expr):
            expr = _core.expr_from_object(expr)
        assert isinstance(expr, _core.Expr)
        sig.return_type = expr.col_type
        if function_path is not None:
            validate_symbol_path(function_path)
        result = ExprTemplateFunction(
            [ExprTemplate(expr, sig)],
            self_path=function_path,
            name=function_name,
            is_method=is_method,
            is_property=is_property,
        )
        if (is_method or is_property) and function_path is not None:
            FunctionRegistry.get().register_function(function_path, result)
        return result

    if len(args) == 1:
        assert len(kwargs) == 0 and callable(args[0])
        return make_expr_template(args[0], None, False, False)
    assert len(args) == 0
    unknown = set(kwargs) - {'param_types', 'is_method', 'is_property'}
    if unknown:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid @expr_udf decorator kwargs: {", ".join(sorted(unknown))}'
        )
    param_types = kwargs.get('param_types')
    is_method = kwargs.get('is_method', False)
    is_property = kwargs.get('is_property', False)
    return lambda py_fn: make_expr_template(py_fn, param_types, is_method, is_property)

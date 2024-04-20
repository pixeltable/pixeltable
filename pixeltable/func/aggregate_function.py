from __future__ import annotations

import abc
import importlib
import inspect
from typing import Optional, Any, Type, List, Dict, Callable
import itertools

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from .function import Function
from .signature import Signature, Parameter
from .globals import validate_symbol_path


class Aggregator(abc.ABC):
    def update(self, *args: Any, **kwargs: Any) -> None:
        pass
    def value(self) -> Any:
        pass


class AggregateFunction(Function):
    """Function interface for an aggregation operation.

    requires_order_by: if True, the first parameter to an aggregate function defines the order in which the function
    sees rows in update()
    allows_std_agg: if True, the aggregate function can be used as a standard aggregate function w/o a window
    allows_window: if True, the aggregate function can be used with a window
    """
    ORDER_BY_PARAM = 'order_by'
    GROUP_BY_PARAM = 'group_by'
    RESERVED_PARAMS = {ORDER_BY_PARAM, GROUP_BY_PARAM}

    def __init__(
            self, aggregator_class: Type[Aggregator], self_path: str,
            init_types: List[ts.ColumnType], update_types: List[ts.ColumnType], value_type: ts.ColumnType,
            requires_order_by: bool, allows_std_agg: bool, allows_window: bool):
        self.agg_cls = aggregator_class
        self.requires_order_by = requires_order_by
        self.allows_std_agg = allows_std_agg
        self.allows_window = allows_window

        # our signature is the signature of 'update', but without self,
        # plus the parameters of 'init' as keyword-only parameters
        update_params = list(inspect.signature(self.agg_cls.update).parameters.values())[1:]  # leave out self
        assert len(update_params) == len(update_types)
        init_params = [
            inspect.Parameter(p.name, inspect.Parameter.KEYWORD_ONLY, default=p.default)
            # starting at 1: leave out self
            for p in itertools.islice(inspect.signature(self.agg_cls.__init__).parameters.values(), 1, None)
        ]
        assert len(init_params) == len(init_types)
        duplicate_params = set(p.name for p in init_params) & set(p.name for p in update_params)
        if len(duplicate_params) > 0:
            raise excs.Error(
                f'__init__() and update() cannot have parameters with the same name: '
                f'{", ".join(duplicate_params)}'
            )
        py_params = update_params + init_params  # init_params are keyword-only and come last
        py_signature = inspect.Signature(py_params)

        params = [Parameter(p.name, update_types[i], p.kind, is_batched=False) for i, p in enumerate(update_params)]
        params.extend([Parameter(p.name, init_types[i], p.kind, is_batched=False) for i, p in enumerate(init_params)])
        signature = Signature(value_type, params)
        super().__init__(signature, py_signature=py_signature, self_path=self_path)
        self.init_param_names = [p.name for p in init_params]

        # make sure the signature doesn't contain reserved parameter names;
        # do this after super().__init__(), otherwise self.name is invalid
        for param in signature.parameters:
            if param.lower() in self.RESERVED_PARAMS:
                raise excs.Error(f'{self.name}(): parameter name {param} is reserved')

    def help_str(self) -> str:
        res = super().help_str()
        res += '\n\n' + inspect.getdoc(self.agg_cls.update)
        return res

    def __call__(self, *args: object, **kwargs: object) -> 'pixeltable.exprs.Expr':
        from pixeltable import exprs

        # perform semantic analysis of special parameters 'order_by' and 'group_by'
        order_by_clause: Optional[Any] = None
        if self.ORDER_BY_PARAM in kwargs:
            if self.requires_order_by:
                raise excs.Error(
                    f'{self.display_name}(): order_by invalid, this function requires the first argument to be the '
                    f'ordering expression'
                )
            if not self.allows_window:
                raise excs.Error(
                    f'{self.display_name}(): order_by invalid with an aggregate function that does not allow windows')
            order_by_clause = kwargs.pop(self.ORDER_BY_PARAM)
        elif self.requires_order_by:
            # the first argument is the order-by expr
            if len(args) == 0:
                raise excs.Error(f'{self.display_name}(): requires an ordering expression as its first argument')
            order_by_clause = args[0]
            if not isinstance(order_by_clause, exprs.Expr):
                raise excs.Error(
                    f'{self.display_name}(): the first argument needs to be a Pixeltable expression, but instead is a '
                    f'{type(order_by_clause)}'
                )
            # don't pass the first parameter on, the Function doesn't get to see it
            args = args[1:]

        group_by_clause: Optional[Any] = None
        if self.GROUP_BY_PARAM in kwargs:
            if not self.allows_window:
                raise excs.Error(
                    f'{self.display_name}(): group_by invalid with an aggregate function that does not allow windows')
            group_by_clause = kwargs.pop(self.GROUP_BY_PARAM)

        bound_args = self.py_signature.bind(*args, **kwargs)
        self.validate_call(bound_args.arguments)
        return exprs.FunctionCall(
            self, bound_args.arguments,
            order_by_clause=[order_by_clause] if order_by_clause is not None else [],
            group_by_clause=[group_by_clause] if group_by_clause is not None else [])

    def validate_call(self, bound_args: Dict[str, Any]) -> None:
        # check that init parameters are not Exprs
        # TODO: do this in the planner (check that init parameters are either constants or only refer to grouping exprs)
        import pixeltable.exprs as exprs
        for param_name in self.init_param_names:
            if param_name in bound_args and isinstance(bound_args[param_name], exprs.Expr):
                raise excs.Error(
                    f'{self.display_name}(): init() parameter {param_name} needs to be a constant, not a Pixeltable '
                    f'expression'
                )


def uda(
        *,
        value_type: ts.ColumnType,
        update_types: List[ts.ColumnType],
        init_types: Optional[List[ts.ColumnType]] = None,
        requires_order_by: bool = False, allows_std_agg: bool = True, allows_window: bool = False,
) -> Callable:
    """Decorator for user-defined aggregate functions.

    The decorated class must inherit from Aggregator and implement the following methods:
    - __init__(self, ...) to initialize the aggregator
    - update(self, ...) to update the aggregator with a new value
    - value(self) to return the final result

    The decorator creates an AggregateFunction instance from the class and adds it
    to the module where the class is defined.

    Parameters:
    - init_types: list of types for the __init__() parameters; must match the number of parameters
    - update_types: list of types for the update() parameters; must match the number of parameters
    - value_type: return type of the aggregator
    - requires_order_by: if True, the first parameter to the function is the order-by expression
    - allows_std_agg: if True, the function can be used as a standard aggregate function w/o a window
    - allows_window: if True, the function can be used with a window
    """
    if init_types is None:
        init_types = []

    def decorator(cls: Type[Aggregator]) -> Type[Function]:
        # validate type parameters
        num_init_params = len(inspect.signature(cls.__init__).parameters) - 1
        if num_init_params > 0:
            if len(init_types) != num_init_params:
                raise excs.Error(
                    f'init_types must be a list of {num_init_params} types, one for each parameter of __init__()')
        num_update_params = len(inspect.signature(cls.update).parameters) - 1
        if num_update_params == 0:
            raise excs.Error('update() must have at least one parameter')
        if len(update_types) != num_update_params:
            raise excs.Error(
                f'update_types must be a list of {num_update_params} types, one for each parameter of update()')
        assert value_type is not None

        # the AggregateFunction instance resides in the same module as cls
        class_path = f'{cls.__module__}.{cls.__qualname__}'
        # nonlocal name
        # name = name or cls.__name__
        # instance_path_elements = class_path.split('.')[:-1] + [name]
        # instance_path = '.'.join(instance_path_elements)

        # create the corresponding AggregateFunction instance
        instance = AggregateFunction(
            cls, class_path, init_types, update_types, value_type, requires_order_by, allows_std_agg, allows_window)
        # do the path validation at the very end, in order to be able to write tests for the other failure cases
        validate_symbol_path(class_path)
        #module = importlib.import_module(cls.__module__)
        #setattr(module, name, instance)

        return instance

    return decorator

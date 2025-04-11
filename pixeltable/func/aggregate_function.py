from __future__ import annotations

import abc
import inspect
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional, Sequence, overload

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .function import Function
from .globals import validate_symbol_path
from .signature import Parameter, Signature

if TYPE_CHECKING:
    from pixeltable import exprs


class Aggregator(abc.ABC):
    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def value(self) -> Any: ...


class AggregateFunction(Function):
    """Function interface for an aggregation operation.

    requires_order_by: if True, the first parameter to an aggregate function defines the order in which the function
    sees rows in update()
    allows_std_agg: if True, the aggregate function can be used as a standard aggregate function w/o a window
    allows_window: if True, the aggregate function can be used with a window
    """

    ORDER_BY_PARAM: ClassVar[str] = 'order_by'
    GROUP_BY_PARAM: ClassVar[str] = 'group_by'
    RESERVED_PARAMS: ClassVar[set[str]] = {ORDER_BY_PARAM, GROUP_BY_PARAM}

    agg_classes: list[type[Aggregator]]  # classes for each signature, in signature order
    init_param_names: list[list[str]]  # names of the __init__ parameters for each signature

    def __init__(
        self,
        agg_class: type[Aggregator],
        type_substitutions: Optional[Sequence[dict]],
        self_path: str,
        requires_order_by: bool,
        allows_std_agg: bool,
        allows_window: bool,
    ) -> None:
        if type_substitutions is None:
            type_substitutions = [None]  # single signature with no substitutions
            self.agg_classes = [agg_class]
        else:
            self.agg_classes = [agg_class] * len(type_substitutions)
        self.init_param_names = []
        self.requires_order_by = requires_order_by
        self.allows_std_agg = allows_std_agg
        self.allows_window = allows_window
        self.__doc__ = agg_class.__doc__

        signatures: list[Signature] = []

        # If no type_substitutions were provided, construct a single signature for the class.
        # Otherwise, construct one signature for each type substitution instance.
        for subst in type_substitutions:
            signature, init_param_names = self.__cls_to_signature(agg_class, subst)
            signatures.append(signature)
            self.init_param_names.append(init_param_names)

        super().__init__(signatures, self_path=self_path)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        self.agg_classes = [self.agg_classes[signature_idx]]
        self.init_param_names = [self.init_param_names[signature_idx]]

    def __cls_to_signature(
        self, cls: type[Aggregator], type_substitutions: Optional[dict] = None
    ) -> tuple[Signature, list[str]]:
        """Inspects the Aggregator class to infer the corresponding function signature. Returns the
        inferred signature along with the list of init_param_names (for downstream error handling).
        """
        from pixeltable import exprs

        # infer type parameters; set return_type=InvalidType() because it has no meaning here
        init_sig = Signature.create(
            py_fn=cls.__init__, return_type=ts.InvalidType(), is_cls_method=True, type_substitutions=type_substitutions
        )
        update_sig = Signature.create(
            py_fn=cls.update, return_type=ts.InvalidType(), is_cls_method=True, type_substitutions=type_substitutions
        )
        value_sig = Signature.create(py_fn=cls.value, is_cls_method=True, type_substitutions=type_substitutions)

        init_types = [p.col_type for p in init_sig.parameters.values()]
        update_types = [p.col_type for p in update_sig.parameters.values()]
        value_type = value_sig.return_type
        assert value_type is not None

        if len(update_types) == 0:
            raise excs.Error('update() must have at least one parameter')

        # our signature is the signature of 'update', but without self,
        # plus the parameters of 'init' as keyword-only parameters
        py_update_params = list(inspect.signature(cls.update).parameters.values())[1:]  # leave out self
        assert len(py_update_params) == len(update_types)
        update_params = [
            Parameter(
                p.name,
                col_type=update_types[i],
                kind=p.kind,
                default=exprs.Expr.from_object(p.default),  # type: ignore[arg-type]
            )
            for i, p in enumerate(py_update_params)
        ]
        # starting at 1: leave out self
        py_init_params = list(inspect.signature(cls.__init__).parameters.values())[1:]
        assert len(py_init_params) == len(init_types)
        init_params = [
            Parameter(
                p.name,
                col_type=init_types[i],
                kind=inspect.Parameter.KEYWORD_ONLY,
                default=exprs.Expr.from_object(p.default),  # type: ignore[arg-type]
            )
            for i, p in enumerate(py_init_params)
        ]
        duplicate_params = {p.name for p in init_params} & {p.name for p in update_params}
        if len(duplicate_params) > 0:
            raise excs.Error(
                f'__init__() and update() cannot have parameters with the same name: {", ".join(duplicate_params)}'
            )
        params = update_params + init_params  # init_params are keyword-only and come last
        init_param_names = [p.name for p in init_params]

        return Signature(value_type, params), init_param_names

    @property
    def agg_class(self) -> type[Aggregator]:
        assert not self.is_polymorphic
        return self.agg_classes[0]

    @property
    def is_async(self) -> bool:
        return False

    def exec(self, args: Sequence[Any], kwargs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def overload(self, cls: type[Aggregator]) -> AggregateFunction:
        if not isinstance(cls, type) or not issubclass(cls, Aggregator):
            raise excs.Error(f'Invalid argument to @overload decorator: {cls}')
        if self._has_resolved_fns:
            raise excs.Error('New `overload` not allowed after the UDF has already been called')
        if self._conditional_return_type is not None:
            raise excs.Error('New `overload` not allowed after a conditional return type has been specified')
        sig, init_param_names = self.__cls_to_signature(cls)
        self.signatures.append(sig)
        self.agg_classes.append(cls)
        self.init_param_names.append(init_param_names)
        return self

    def comment(self) -> Optional[str]:
        return inspect.getdoc(self.agg_classes[0])

    def help_str(self) -> str:
        res = super().help_str()
        # We need to reference agg_classes[0] rather than agg_class here, because we want this to work even if the
        # aggregator is polymorphic (in which case we use the docstring of the originally decorated UDA).
        res += '\n\n' + inspect.getdoc(self.agg_classes[0].update)
        return res

    def __call__(self, *args: Any, **kwargs: Any) -> 'exprs.FunctionCall':
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
                    f'{self.display_name}(): order_by invalid with an aggregate function that does not allow windows'
                )
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
                    f'{self.display_name}(): group_by invalid with an aggregate function that does not allow windows'
                )
            group_by_clause = kwargs.pop(self.GROUP_BY_PARAM)

        args = [exprs.Expr.from_object(arg) for arg in args]
        kwargs = {k: exprs.Expr.from_object(v) for k, v in kwargs.items()}

        resolved_fn, bound_args = self._bind_to_matching_signature(args, kwargs)
        return_type = resolved_fn.call_return_type(bound_args)

        return exprs.FunctionCall(
            resolved_fn,
            args,
            kwargs,
            return_type,
            order_by_clause=[order_by_clause] if order_by_clause is not None else [],
            group_by_clause=[group_by_clause] if group_by_clause is not None else [],
        )

    def validate_call(self, bound_args: dict[str, 'exprs.Expr']) -> None:
        from pixeltable import exprs

        super().validate_call(bound_args)

        # check that init parameters are not Exprs
        # TODO: do this in the planner (check that init parameters are either constants or only refer to grouping exprs)
        for param_name in self.init_param_names[0]:
            if param_name in bound_args and not isinstance(bound_args[param_name], exprs.Literal):
                raise excs.Error(f'{self.display_name}(): init() parameter {param_name!r} must be a constant value')

    def __repr__(self) -> str:
        return f'<Pixeltable Aggregator {self.name}>'


# Decorator invoked without parentheses: @pxt.uda
@overload
def uda(decorated_fn: Callable) -> AggregateFunction: ...


# Decorator schema invoked with parentheses: @pxt.uda(**kwargs)
@overload
def uda(
    *,
    requires_order_by: bool = False,
    allows_std_agg: bool = True,
    allows_window: bool = False,
    type_substitutions: Optional[Sequence[dict]] = None,
) -> Callable[[type[Aggregator]], AggregateFunction]: ...


def uda(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Decorator for user-defined aggregate functions.

    The decorated class must inherit from Aggregator and implement the following methods:
    - __init__(self, ...) to initialize the aggregator
    - update(self, ...) to update the aggregator with a new value
    - value(self) to return the final result

    The decorator creates an AggregateFunction instance from the class and adds it
    to the module where the class is defined.

    Parameters:
    - requires_order_by: if True, the first parameter to the function is the order-by expression
    - allows_std_agg: if True, the function can be used as a standard aggregate function w/o a window
    - allows_window: if True, the function can be used with a window
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # Decorator invoked without parentheses: @pxt.uda
        # Simply call make_aggregator with defaults.
        return make_aggregator(cls=args[0])

    else:
        # Decorator schema invoked with parentheses: @pxt.uda(**kwargs)
        # Create a decorator for the specified schema.
        requires_order_by = kwargs.pop('requires_order_by', False)
        allows_std_agg = kwargs.pop('allows_std_agg', True)
        allows_window = kwargs.pop('allows_window', False)
        type_substitutions = kwargs.pop('type_substitutions', None)
        if len(kwargs) > 0:
            raise excs.Error(f'Invalid @uda decorator kwargs: {", ".join(kwargs.keys())}')
        if len(args) > 0:
            raise excs.Error('Unexpected @uda decorator arguments.')

        def decorator(cls: type[Aggregator]) -> AggregateFunction:
            return make_aggregator(
                cls,
                requires_order_by=requires_order_by,
                allows_std_agg=allows_std_agg,
                allows_window=allows_window,
                type_substitutions=type_substitutions,
            )

        return decorator


def make_aggregator(
    cls: type[Aggregator],
    requires_order_by: bool = False,
    allows_std_agg: bool = True,
    allows_window: bool = False,
    type_substitutions: Optional[Sequence[dict]] = None,
) -> AggregateFunction:
    class_path = f'{cls.__module__}.{cls.__qualname__}'
    instance = AggregateFunction(cls, type_substitutions, class_path, requires_order_by, allows_std_agg, allows_window)
    # do the path validation at the very end, in order to be able to write tests for the other failure cases
    validate_symbol_path(class_path)
    return instance

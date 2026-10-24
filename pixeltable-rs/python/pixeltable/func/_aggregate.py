"""User-defined aggregate functions (`@pxt.uda`), upstream `pixeltable.func.aggregate_function`.

Python introspects the `Aggregator` subclass (signatures from `update()` and `__init__()`); the backend builds the
aggregate calls and drives the aggregation, calling `__init__()`, `update()` and `value()` of the class.
"""

from __future__ import annotations

import abc
import inspect
from typing import Any, ClassVar, Sequence

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from ._function import Function, validate_symbol_path
from ._signature import Parameter, Signature, _literal


class Aggregator(abc.ABC):
    """Base class of user-defined aggregators (`@pxt.uda`)."""

    @abc.abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None: ...

    @abc.abstractmethod
    def value(self) -> Any: ...


def _default_expr(p: inspect.Parameter) -> Any:
    if p.default is inspect.Parameter.empty:
        return None
    return _literal(p.default)


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

    agg_classes: list[type[Aggregator]]
    init_param_names: list[list[str]]

    def __init__(
        self,
        agg_class: type[Aggregator],
        type_substitutions: Sequence[dict] | None,
        self_path: str,
        requires_order_by: bool,
        allows_std_agg: bool,
        allows_window: bool,
    ) -> None:
        substs: Sequence[dict | None]
        if type_substitutions is None:
            substs = [None]
            self.agg_classes = [agg_class]
        else:
            substs = type_substitutions
            self.agg_classes = [agg_class] * len(type_substitutions)
        self.init_param_names = []
        self.requires_order_by = requires_order_by
        self.allows_std_agg = allows_std_agg
        self.allows_window = allows_window
        self.__doc__ = agg_class.__doc__

        signatures: list[Signature] = []
        for subst in substs:
            signature, init_param_names = self.__cls_to_signature(agg_class, subst)
            signatures.append(signature)
            self.init_param_names.append(init_param_names)
        super().__init__(signatures, self_path=self_path)

    def _update_as_overload_resolution(self, signature_idx: int) -> None:
        self.agg_classes = [self.agg_classes[signature_idx]]
        self.init_param_names = [self.init_param_names[signature_idx]]

    def __cls_to_signature(
        self, cls: type[Aggregator], type_substitutions: dict | None = None
    ) -> tuple[Signature, list[str]]:
        """The signature of `update()` (without `self`) followed by the parameters of `__init__()` as keyword-only
        parameters, and the names of the latter."""
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
        if len(update_types) == 0:
            raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, 'update() must have at least one parameter')

        py_update_params = list(inspect.signature(cls.update, eval_str=True).parameters.values())[1:]
        assert len(py_update_params) == len(update_types)
        update_params = [
            Parameter(p.name, col_type=update_types[i], kind=p.kind, default=_default_expr(p))
            for i, p in enumerate(py_update_params)
        ]
        py_init_params = list(inspect.signature(cls.__init__, eval_str=True).parameters.values())[1:]
        assert len(py_init_params) == len(init_types)
        init_params = [
            Parameter(p.name, col_type=init_types[i], kind=inspect.Parameter.KEYWORD_ONLY, default=_default_expr(p))
            for i, p in enumerate(py_init_params)
        ]
        duplicate_params = {p.name for p in init_params} & {p.name for p in update_params}
        if len(duplicate_params) > 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION,
                f'__init__() and update() cannot have parameters with the same name: {", ".join(duplicate_params)}',
            )
        return Signature(value_type, update_params + init_params), [p.name for p in init_params]

    @property
    def agg_class(self) -> type[Aggregator]:
        assert not self.is_polymorphic
        return self.agg_classes[0]

    @property
    def is_async(self) -> bool:
        return False

    @property
    def is_aggregate(self) -> bool:
        return True

    def overload(self, cls: type[Aggregator]) -> AggregateFunction:
        if not isinstance(cls, type) or not issubclass(cls, Aggregator):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid argument to @overload decorator: {cls}'
            )
        if self._has_resolved_fns or self._has_core:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE, 'New `overload` not allowed after the UDF has already been called'
            )
        if self._conditional_return_type is not None:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_STATE,
                'New `overload` not allowed after a conditional return type has been specified',
            )
        sig, init_param_names = self.__cls_to_signature(cls)
        self.signatures.append(sig)
        self.agg_classes.append(cls)
        self.init_param_names.append(init_param_names)
        return self

    def comment(self) -> str | None:
        return inspect.getdoc(self.agg_classes[0])

    def help_str(self) -> str:
        res = super().help_str()
        doc = inspect.getdoc(self.agg_classes[0].update)
        return res if doc is None else f'{res}\n\n{doc}'

    def _core_spec(self) -> dict[str, Any]:
        return {
            'kind': 'aggregate',
            'name': self.name,
            'display_name': self.display_name,
            'self_path': self.self_path,
            'signatures': [sig._to_core() for sig in self.signatures],
            'agg_classes': self.agg_classes,
            'init_param_names': self.init_param_names,
            'requires_order_by': self.requires_order_by,
            'allows_std_agg': self.allows_std_agg,
            'allows_window': self.allows_window,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f'calling the aggregate function `{self.name}` is not implemented yet in pixeltable-rs'
        )

    def __repr__(self) -> str:
        return f'<Pixeltable Aggregator {self.name}>'


def uda(*args: Any, **kwargs: Any) -> Any:
    """Decorator for user-defined aggregate functions.

    The decorated class must inherit from Aggregator and implement:
    - __init__(self, ...) to initialize the aggregator
    - update(self, ...) to update the aggregator with a new value
    - value(self) to return the final result

    Parameters:
    - requires_order_by: if True, the first parameter to the function is the order-by expression
    - allows_std_agg: if True, the function can be used as a standard aggregate function w/o a window
    - allows_window: if True, the function can be used with a window
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return make_aggregator(cls=args[0])
    requires_order_by = kwargs.pop('requires_order_by', False)
    allows_std_agg = kwargs.pop('allows_std_agg', True)
    allows_window = kwargs.pop('allows_window', False)
    type_substitutions = kwargs.pop('type_substitutions', None)
    if len(kwargs) > 0:
        raise excs.RequestError(
            excs.ErrorCode.INVALID_CONFIGURATION, f'Invalid @uda decorator kwargs: {", ".join(kwargs.keys())}'
        )
    if len(args) > 0:
        raise excs.RequestError(excs.ErrorCode.INVALID_CONFIGURATION, 'Unexpected @uda decorator arguments.')

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
    type_substitutions: Sequence[dict] | None = None,
) -> AggregateFunction:
    class_path = f'{cls.__module__}.{cls.__qualname__}'
    instance = AggregateFunction(cls, type_substitutions, class_path, requires_order_by, allows_std_agg, allows_window)
    validate_symbol_path(class_path)
    return instance



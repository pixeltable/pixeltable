from __future__ import annotations

import inspect
import sys
from typing import Any, Optional, Sequence
from uuid import UUID

import sqlalchemy as sql
from typing_extensions import Self

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.func as func
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .rowid_ref import RowidRef
from .sql_element_cache import SqlElementCache


class FunctionCall(Expr):
    fn: func.Function
    is_method_call: bool
    agg_init_args: dict[str, Any]
    resource_pool: Optional[str]

    normalized_args: list[int]
    normalized_kwargs: dict[str, int]

    # maps each parameter name to tuple representing the value it has in the call:
    # - argument's index in components, if an argument is given in the call
    # - default value, if no argument given in the call
    # (in essence, this combines init()'s bound_args and default values)
    _param_values: dict[str, tuple[Optional[int], Optional[Any]]]

    return_type: ts.ColumnType
    group_by_start_idx: int
    group_by_stop_idx: int
    fn_expr_idx: int
    order_by_idx: int
    aggregator: Optional[Any]
    current_partition_vals: Optional[list[Any]]
    original_args: list[Expr]
    original_kwargs: dict[str, Expr]

    def __init__(
        self,
        fn: func.Function,
        bound_args: dict[str, Expr],
        return_type: ts.ColumnType,
        order_by_clause: Optional[list[Any]] = None,
        group_by_clause: Optional[list[Any]] = None,
        is_method_call: bool = False,
        original_args: list[Expr] = None,
        original_kwargs: dict[str, Expr] = None,
    ):
        assert all(isinstance(arg, Expr) for arg in bound_args.values())
        assert all(isinstance(arg, Expr) for arg in original_args)
        assert all(isinstance(arg, Expr) for arg in original_kwargs.values())

        self.original_args = original_args
        self.original_kwargs = original_kwargs

        if order_by_clause is None:
            order_by_clause = []
        if group_by_clause is None:
            group_by_clause = []

        assert not fn.is_polymorphic

        self.fn = fn
        self.is_method_call = is_method_call
        self.resource_pool = fn.call_resource_pool(bound_args)
        signature = fn.signature

        # If `return_type` is non-nullable, but the function call has a nullable input to any of its non-nullable
        # parameters, then we need to make it nullable. This is because Pixeltable defaults a function output to
        # `None` when any of its non-nullable inputs are `None`.
        for arg_name, arg in bound_args.items():
            param = signature.parameters[arg_name]
            if param.col_type is not None and not param.col_type.nullable and arg.col_type.nullable:
                return_type = return_type.copy(nullable=True)
                break

        self.return_type = return_type

        super().__init__(return_type)

        self.agg_init_args = {}
        if self.is_agg_fn_call:
            # We separate out the init args for the aggregator. Unpack Literals in init args.
            assert isinstance(fn, func.AggregateFunction)
            for arg_name, arg in bound_args.items():
                if arg_name in fn.init_param_names[0]:
                    assert isinstance(arg, Literal)  # This was checked during validate_call
                    self.agg_init_args[arg_name] = arg.val
            bound_args = {
                arg_name: arg for arg_name, arg in bound_args.items() if arg_name not in fn.init_param_names[0]
            }

        # construct components, args, kwargs
        self.normalized_args = []
        self.normalized_kwargs = {}
        self._param_values = {}

        # the prefix of parameters that are bound can be passed by position
        processed_args: set[str] = set()
        for py_param in signature.py_signature.parameters.values():
            if py_param.name not in bound_args or py_param.kind == inspect.Parameter.KEYWORD_ONLY:
                break
            arg = bound_args[py_param.name]
            self.normalized_args.append(len(self.components))
            self._param_values[py_param.name] = (len(self.components), None)
            self.components.append(arg.copy())
            processed_args.add(py_param.name)

        # the remaining args are passed as keywords
        for param_name in bound_args.keys():
            if param_name not in processed_args:
                arg = bound_args[param_name]
                self.normalized_kwargs[param_name] = len(self.components)
                self._param_values[param_name] = (len(self.components), None)
                self.components.append(arg.copy())

        # fill in default values for parameters that don't have explicit arguments
        for param in fn.signature.parameters.values():
            if param.name not in self._param_values:
                self._param_values[param.name] = (None, None if param.default is None else param.default.val)

        # window function state:
        # self.components[self.group_by_start_idx:self.group_by_stop_idx] contains group_by exprs
        self.group_by_start_idx, self.group_by_stop_idx = 0, 0
        if len(group_by_clause) > 0:
            if isinstance(group_by_clause[0], catalog.Table):
                assert len(group_by_clause) == 1
                group_by_exprs = self._create_rowid_refs(group_by_clause[0])
            else:
                assert all(isinstance(expr, Expr) for expr in group_by_clause)
                group_by_exprs = group_by_clause
            # record grouping exprs in self.components, we need to evaluate them to get partition vals
            self.group_by_start_idx = len(self.components)
            self.group_by_stop_idx = len(self.components) + len(group_by_exprs)
            self.components.extend(group_by_exprs)

        if isinstance(self.fn, func.ExprTemplateFunction):
            # we instantiate the template to create an Expr that can be evaluated and record that as a component
            fn_expr = self.fn.instantiate([], bound_args)
            self.fn_expr_idx = len(self.components)
            self.components.append(fn_expr)
        else:
            self.fn_expr_idx = sys.maxsize

        # we want to make sure that order_by_clause get assigned slot_idxs, even though we won't need to evaluate them
        # (that's done in SQL)
        if len(order_by_clause) > 0 and not isinstance(order_by_clause[0], Expr):
            raise excs.Error(
                f'order_by argument needs to be a Pixeltable expression, but instead is a {type(order_by_clause[0])}'
            )
        # don't add components after this, everthing after order_by_start_idx is part of the order_by clause
        self.order_by_idx = len(self.components)
        self.components.extend(order_by_clause)

        # execution state for aggregate functions
        self.aggregator = None
        self.current_partition_vals = None

        self.id = self._create_id()

    def _create_rowid_refs(self, tbl: catalog.Table) -> list[Expr]:
        target = tbl._tbl_version_path.tbl_version
        return [RowidRef(target, i) for i in range(target.num_rowid_columns())]

    def default_column_name(self) -> Optional[str]:
        return self.fn.name

    def _equals(self, other: FunctionCall) -> bool:
        if self.fn != other.fn:
            return False
        if len(self.normalized_args) != len(other.normalized_args):
            return False
        for i in range(len(self.normalized_args)):
            if self.normalized_args[i] != other.normalized_args[i]:
                return False
        if self.group_by_start_idx != other.group_by_start_idx:
            return False
        if self.group_by_stop_idx != other.group_by_stop_idx:
            return False
        if self.order_by_idx != other.order_by_idx:
            return False
        return True

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [
            ('fn', id(self.fn)),  # use the function pointer, not the fqn, which isn't set for lambdas
            ('args', self.normalized_args),
            ('kwargs', self.normalized_kwargs),
            ('group_by_start_idx', self.group_by_start_idx),
            ('group_by_stop_idx', self.group_by_stop_idx),
            ('fn_expr_idx', self.fn_expr_idx),
            ('order_by_idx', self.order_by_idx),
        ]

    def __repr__(self) -> str:
        return self.display_str()

    def display_str(self, inline: bool = True) -> str:
        if self.is_method_call:
            return f'{self.components[0]}.{self.fn.name}({self._print_args(1, inline)})'
        else:
            fn_name = self.fn.display_name if self.fn.display_name != '' else 'anonymous_fn'
            return f'{fn_name}({self._print_args()})'

    def _print_args(self, start_idx: int = 0, inline: bool = True) -> str:
        def print_arg(arg: Any) -> str:
            return repr(arg) if isinstance(arg, str) else str(arg)

        arg_strs = [str(self.components[idx]) for idx in self.normalized_args[start_idx:]]
        arg_strs.extend(
            [f'{param_name}={str(self.components[idx])}' for param_name, idx in self.normalized_kwargs.items()]
        )
        if len(self.order_by) > 0:
            assert isinstance(self.fn, func.AggregateFunction)
            if self.fn.requires_order_by:
                arg_strs.insert(0, Expr.print_list(self.order_by))
            else:
                arg_strs.append(f'order_by={Expr.print_list(self.order_by)}')
        if len(self.group_by) > 0:
            arg_strs.append(f'group_by={Expr.print_list(self.group_by)}')
        # TODO: figure out the function name
        separator = ', ' if inline else ',\n    '
        return separator.join(arg_strs)

    def has_group_by(self) -> bool:
        return self.group_by_stop_idx != 0

    @property
    def group_by(self) -> list[Expr]:
        return self.components[self.group_by_start_idx : self.group_by_stop_idx]

    @property
    def order_by(self) -> list[Expr]:
        return self.components[self.order_by_idx :]

    @property
    def is_window_fn_call(self) -> bool:
        return (
            isinstance(self.fn, func.AggregateFunction)
            and self.fn.allows_window
            and (
                not self.fn.allows_std_agg
                or self.has_group_by()
                or (len(self.order_by) > 0 and not self.fn.requires_order_by)
            )
        )

    def get_window_sort_exprs(self) -> tuple[list[Expr], list[Expr]]:
        return self.group_by, self.order_by

    def get_window_ordering(self) -> list[tuple[Expr, bool]]:
        # ordering is implicitly ascending
        return [(e, None) for e in self.group_by] + [(e, True) for e in self.order_by]

    @property
    def is_agg_fn_call(self) -> bool:
        return isinstance(self.fn, func.AggregateFunction)

    def get_agg_order_by(self) -> list[Expr]:
        assert self.is_agg_fn_call
        return self.order_by

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        # we currently can't translate aggregate functions with grouping and/or ordering to SQL
        if self.has_group_by() or len(self.order_by) > 0:
            return None

        # try to construct args and kwargs to call self.fn._to_sql()
        kwargs: dict[str, sql.ColumnElement] = {}
        for param_name, component_idx in self.normalized_kwargs.items():
            param = self.fn.signature.parameters[param_name]
            assert param.kind != inspect.Parameter.VAR_POSITIONAL and param.kind != inspect.Parameter.VAR_KEYWORD
            arg_element = sql_elements.get(self.components[component_idx])
            if arg_element is None:
                return None
            kwargs[param_name] = arg_element

        args: list[sql.ColumnElement] = []
        for component_idx in self.normalized_args:
            arg_element = sql_elements.get(self.components[component_idx])
            if arg_element is None:
                return None
            args.append(arg_element)
        result = self.fn._to_sql(*args, **kwargs)

        return result

    def reset_agg(self) -> None:
        """
        Init agg state
        """
        assert self.is_agg_fn_call
        assert isinstance(self.fn, func.AggregateFunction)
        self.aggregator = self.fn.agg_class(**self.agg_init_args)

    def update(self, data_row: DataRow) -> None:
        """
        Update agg state
        """
        assert self.is_agg_fn_call
        args, kwargs = self.make_args(data_row)
        self.aggregator.update(*args, **kwargs)

    def make_args(self, data_row: DataRow) -> Optional[tuple[list[Any], dict[str, Any]]]:
        """Return args and kwargs, constructed for data_row; returns None if any non-nullable arg is None."""
        kwargs: dict[str, Any] = {}
        for param_name, component_idx in self.normalized_kwargs.items():
            val = data_row[self.components[component_idx].slot_idx]
            param = self.fn.signature.parameters[param_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # expand **kwargs parameter
                kwargs.update(val)
            else:
                assert param.kind != inspect.Parameter.VAR_POSITIONAL
                if not param.col_type.nullable and val is None:
                    return None
                kwargs[param_name] = val

        args: list[Any] = []
        for param_idx, component_idx in enumerate(self.normalized_args):
            val = data_row[self.components[component_idx].slot_idx]
            param = self.fn.signature.parameters_by_pos[param_idx]
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                # expand *args parameter
                assert isinstance(val, list)
                args.extend(val)
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                # expand **kwargs parameter
                assert isinstance(val, dict)
                kwargs.update(val)
            else:
                if not param.col_type.nullable and val is None:
                    return None
                args.append(val)

        return args, kwargs

    def get_param_values(self, param_names: Sequence[str], data_rows: list[DataRow]) -> list[dict[str, Any]]:
        """
        Returns a list of dicts mapping each param name to its value when this FunctionCall is evaluated against
        data_rows
        """
        assert all(name in self._param_values for name in param_names), f'{param_names}, {self._param_values.keys()}'
        result: list[dict[str, Any]] = []
        for row in data_rows:
            d: dict[str, Any] = {}
            for param_name in param_names:
                component_idx, default_val = self._param_values[param_name]
                if component_idx is None:
                    d[param_name] = default_val
                else:
                    slot_idx = self.components[component_idx].slot_idx
                    d[param_name] = row[slot_idx]
            result.append(d)
        return result

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        if isinstance(self.fn, func.ExprTemplateFunction):
            # we need to evaluate the template
            # TODO: can we get rid of this extra copy?
            fn_expr = self.components[self.fn_expr_idx]
            data_row[self.slot_idx] = data_row[fn_expr.slot_idx]
            return
        elif self.is_agg_fn_call and not self.is_window_fn_call:
            if self.aggregator is None:
                pass
            data_row[self.slot_idx] = self.aggregator.value()
            return

        args_kwargs = self.make_args(data_row)
        if args_kwargs is None:
            # we can't evaluate this function
            data_row[self.slot_idx] = None
            return
        args, kwargs = args_kwargs

        if isinstance(self.fn, func.CallableFunction) and not self.fn.is_batched:
            # optimization: avoid additional level of indirection we'd get from calling Function.exec()
            data_row[self.slot_idx] = self.fn.py_fn(*args, **kwargs)
        elif self.is_window_fn_call:
            assert isinstance(self.fn, func.AggregateFunction)
            agg_cls = self.fn.agg_class
            if self.has_group_by():
                if self.current_partition_vals is None:
                    self.current_partition_vals = [None] * len(self.group_by)
                partition_vals = [data_row[e.slot_idx] for e in self.group_by]
                if partition_vals != self.current_partition_vals:
                    # new partition
                    self.aggregator = agg_cls(**self.agg_init_args)
                    self.current_partition_vals = partition_vals
            elif self.aggregator is None:
                self.aggregator = agg_cls(**self.agg_init_args)
            self.aggregator.update(*args)
            data_row[self.slot_idx] = self.aggregator.value()
        else:
            data_row[self.slot_idx] = self.fn.exec(args, kwargs)

    def _retarget(self, tbl_versions: dict[UUID, catalog.TableVersion]) -> Self:
        super()._retarget(tbl_versions)
        for i in range(len(self.original_args)):
            self.original_args[i] = self.original_args[i]._retarget(tbl_versions)
        for k in self.original_kwargs:
            self.original_kwargs[k] = self.original_kwargs[k]._retarget(tbl_versions)
        return self

    def _as_dict(self) -> dict:
        group_by_exprs = self.components[self.group_by_start_idx : self.group_by_stop_idx]
        order_by_exprs = self.components[self.order_by_idx :]
        return {
            'fn': self.fn.as_dict(),
            'return_type': self.return_type.as_dict(),
            'group_by_exprs': [expr.as_dict() for expr in group_by_exprs],
            'order_by_exprs': [expr.as_dict() for expr in order_by_exprs],
            'is_method_call': self.is_method_call,
            'args': [expr.as_dict() for expr in self.original_args],
            'kwargs': {name: expr.as_dict() for name, expr in self.original_kwargs.items()},
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> FunctionCall:
        fn = func.Function.from_dict(d['fn'])
        return_type = ts.ColumnType.from_dict(d['return_type']) if 'return_type' in d else None
        group_by_exprs = [Expr.from_dict(expr_d) for expr_d in d['group_by_exprs']]
        order_by_exprs = [Expr.from_dict(expr_d) for expr_d in d['order_by_exprs']]
        is_method_call = d['is_method_call']
        args = [Expr.from_dict(expr_d) for expr_d in d['args']]
        kwargs = {name: Expr.from_dict(expr_d) for name, expr_d in d['kwargs'].items()}

        # Now re-bind args and kwargs using the version of `fn` that is currently represented in code. This ensures
        # that we get a valid binding even if the signatures of `fn` have changed since the FunctionCall was
        # serialized.

        resolved_fn: func.Function
        bound_args: dict[str, Expr]

        try:
            resolved_fn, bound_args = fn._bind_to_matching_signature(args, kwargs)
        except (TypeError, excs.Error):
            # TODO: Handle this more gracefully (instead of failing the DB load, allow the DB load to succeed, but
            #       mark any enclosing FunctionCall as unusable). It's the same issue as dealing with a renamed UDF or
            #       FunctionCall return type mismatch.
            signature_note_str = 'any of its signatures' if fn.is_polymorphic else 'its signature'
            instance_signature_str = f'{len(fn.signatures)} signatures' if fn.is_polymorphic else str(fn.signature)
            raise excs.Error(
                f'The signature stored in the database for the UDF `{fn.self_path}` no longer matches '
                f'{signature_note_str} as currently defined in the code.\nThis probably means that the code for '
                f'`{fn.self_path}` has changed in a backward-incompatible way.\n'
                f'Signature in database: {fn}\n'
                f'Signature as currently defined in code: {instance_signature_str}'
            )

        # Evaluate the call_return_type as defined in the current codebase.
        call_return_type = resolved_fn.call_return_type(bound_args)

        if return_type is None:
            # Schema versions prior to 25 did not store the return_type in metadata, and there is no obvious way to
            # infer it during DB migration, so we might encounter a stored return_type of None. In that case, we use
            # the call_return_type that we just inferred (which matches the deserialization behavior prior to
            # version 25).
            return_type = call_return_type
        else:
            # There is a return_type stored in metadata (schema version >= 25).
            # Check that the stored return_type of the UDF call matches the column type of the FunctionCall, and
            # fail-fast if it doesn't (otherwise we risk getting downstream database errors).
            # TODO: Handle this more gracefully (as noted above).
            if not return_type.is_supertype_of(call_return_type, ignore_nullable=True):
                raise excs.Error(
                    f'The return type stored in the database for a UDF call to `{fn.self_path}` no longer matches the '
                    f'return type of the UDF as currently defined in the code.\nThis probably means that the code for '
                    f'`{fn.self_path}` has changed in a backward-incompatible way.\n'
                    f'Return type in database: `{return_type}`\n'
                    f'Return type as currently defined in code: `{call_return_type}`'
                )

        fn_call = cls(
            resolved_fn,
            bound_args,
            return_type,
            group_by_clause=group_by_exprs,
            order_by_clause=order_by_exprs,
            is_method_call=is_method_call,
            original_args=args,
            original_kwargs=kwargs,
        )

        return fn_call

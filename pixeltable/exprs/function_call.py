from __future__ import annotations

import inspect
import sys
from typing import Any, Optional, Sequence, Union

import sqlalchemy as sql

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

    arg_idxs: list[int]
    kwarg_idxs: dict[str, int]
    bound_idxs: dict[str, Union[int, list[int], dict[str, int]]]

    return_type: ts.ColumnType
    group_by_start_idx: int
    group_by_stop_idx: int
    fn_expr_idx: int
    order_by_start_idx: int
    aggregator: Optional[Any]
    current_partition_vals: Optional[list[Any]]

    def __init__(
        self,
        fn: func.Function,
        args: list[Expr],
        kwargs: dict[str, Expr],
        return_type: ts.ColumnType,
        order_by_clause: Optional[list[Any]] = None,
        group_by_clause: Optional[list[Any]] = None,
        is_method_call: bool = False,
    ):
        assert not fn.is_polymorphic
        assert all(isinstance(arg, Expr) for arg in args)
        assert all(isinstance(arg, Expr) for arg in kwargs.values())

        if order_by_clause is None:
            order_by_clause = []
        if group_by_clause is None:
            group_by_clause = []

        super().__init__(return_type)

        self.fn = fn
        self.return_type = return_type
        self.is_method_call = is_method_call

        # Build the components list from the specified args and kwargs, and note the component_idx of each argument.
        self.components.extend(arg.copy() for arg in args)
        self.arg_idxs = list(range(len(self.components)))
        self.components.extend(arg.copy() for arg in kwargs.values())
        self.kwarg_idxs = {name: i + len(args) for i, name in enumerate(kwargs.keys())}

        # Now generate bound_idxs for the args and kwargs indices.
        # This is guaranteed to work, because at this point the call has already been validated.
        # These will be used later to dereference specific parameter values.
        bindings = fn.signature.py_signature.bind(*self.arg_idxs, **self.kwarg_idxs)
        self.bound_idxs = bindings.arguments

        # Sepearately generate bound_args for purposes of determining the resource pool.
        bindings = fn.signature.py_signature.bind(*args, **kwargs)
        bound_args = bindings.arguments
        self.resource_pool = fn.call_resource_pool(bound_args)

        self.agg_init_args = {}
        if self.is_agg_fn_call:
            # We separate out the init args for the aggregator. Unpack Literals in init args.
            assert isinstance(fn, func.AggregateFunction)
            for arg_name, arg in bound_args.items():
                if arg_name in fn.init_param_names[0]:
                    assert isinstance(arg, Literal)  # This was checked during validate_call
                    self.agg_init_args[arg_name] = arg.val

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
            fn_expr = self.fn.instantiate(args, kwargs)
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
        self.order_by_start_idx = len(self.components)
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
        if len(self.arg_idxs) != len(other.arg_idxs):
            return False
        for i in range(len(self.arg_idxs)):
            if self.arg_idxs[i] != other.arg_idxs[i]:
                return False
        if self.group_by_start_idx != other.group_by_start_idx:
            return False
        if self.group_by_stop_idx != other.group_by_stop_idx:
            return False
        if self.order_by_start_idx != other.order_by_start_idx:
            return False
        return True

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [
            ('fn', id(self.fn)),  # use the function pointer, not the fqn, which isn't set for lambdas
            ('args', self.arg_idxs),
            ('kwargs', self.kwarg_idxs),
            ('group_by_start_idx', self.group_by_start_idx),
            ('group_by_stop_idx', self.group_by_stop_idx),
            ('fn_expr_idx', self.fn_expr_idx),
            ('order_by_idx', self.order_by_start_idx),
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
        arg_strs = [str(self.components[idx]) for idx in self.arg_idxs[start_idx:]]
        arg_strs.extend([f'{param_name}={str(self.components[idx])}' for param_name, idx in self.kwarg_idxs.items()])
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
        return self.components[self.order_by_start_idx :]

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
        args: list[sql.ColumnElement] = []
        for component_idx in self.arg_idxs:
            arg_element = sql_elements.get(self.components[component_idx])
            if arg_element is None:
                return None
            args.append(arg_element)

        kwargs: dict[str, sql.ColumnElement] = {}
        for param_name, component_idx in self.kwarg_idxs.items():
            arg_element = sql_elements.get(self.components[component_idx])
            if arg_element is None:
                return None
            kwargs[param_name] = arg_element

        return self.fn._to_sql(*args, **kwargs)

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
        args: list[Any] = []
        parameters_by_pos = self.fn.signature.parameters_by_pos
        for idx in self.arg_idxs:
            val = data_row[self.components[idx].slot_idx]
            if (
                val is None
                and idx < len(parameters_by_pos)
                and parameters_by_pos[idx].kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and not parameters_by_pos[idx].col_type.nullable
            ):
                return None
            args.append(val)

        kwargs: dict[str, Any] = {}
        parameters = self.fn.signature.parameters
        for param_name, idx in self.kwarg_idxs.items():
            val = data_row[self.components[idx].slot_idx]
            if (
                val is None
                and param_name in parameters
                and parameters[param_name].kind
                in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and not parameters[param_name].col_type.nullable
            ):
                return None
            kwargs[param_name] = val

        return args, kwargs

    def get_param_values(self, param_names: Sequence[str], data_rows: list[DataRow]) -> list[dict[str, Any]]:
        """
        Returns a list of dicts mapping each param name to its value when this FunctionCall is evaluated against
        data_rows
        """
        assert all(name in self.fn.signature.parameters for name in param_names), f'{param_names}, {self.fn.signature}'
        result: list[dict[str, Any]] = []
        for row in data_rows:
            d: dict[str, Any] = {}
            for param_name in param_names:
                val = self.bound_idxs.get(param_name)
                if isinstance(val, int):
                    d[param_name] = row[self.components[val].slot_idx]
                elif isinstance(val, list):
                    # var_positional
                    d[param_name] = [row[self.components[idx].slot_idx] for idx in val]
                elif isinstance(val, dict):
                    # var_keyword
                    d[param_name] = {k: row[self.components[idx].slot_idx] for k, idx in val.items()}
                else:
                    assert val is None
                    default = self.fn.signature.parameters[param_name].default
                    assert default is not None
                    d[param_name] = default.val
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

    def _as_dict(self) -> dict:
        return {
            'fn': self.fn.as_dict(),
            'return_type': self.return_type.as_dict(),
            'arg_idxs': self.arg_idxs,
            'kwarg_idxs': self.kwarg_idxs,
            'group_by_start_idx': self.group_by_start_idx,
            'group_by_stop_idx': self.group_by_stop_idx,
            'order_by_start_idx': self.order_by_start_idx,
            'is_method_call': self.is_method_call,
            **super()._as_dict(),
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> FunctionCall:
        fn = func.Function.from_dict(d['fn'])
        return_type = ts.ColumnType.from_dict(d['return_type']) if 'return_type' in d else None
        arg_idxs: list[int] = d['arg_idxs']
        kwarg_idxs: dict[str, int] = d['kwarg_idxs']
        group_by_start_idx: int = d['group_by_start_idx']
        group_by_stop_idx: int = d['group_by_stop_idx']
        order_by_start_idx: int = d['order_by_start_idx']
        is_method_call: bool = d['is_method_call']

        args = [components[idx] for idx in arg_idxs]
        kwargs = {name: components[idx] for name, idx in kwarg_idxs.items()}
        group_by_exprs = components[group_by_start_idx:group_by_stop_idx]
        order_by_exprs = components[order_by_start_idx:]

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
            args,
            kwargs,
            return_type,
            group_by_clause=group_by_exprs,
            order_by_clause=order_by_exprs,
            is_method_call=is_method_call,
        )

        return fn_call

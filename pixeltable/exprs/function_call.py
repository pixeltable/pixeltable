from __future__ import annotations

import inspect
import json
import sys
from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.func as func
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .inline_expr import InlineDict, InlineList
from .row_builder import RowBuilder
from .rowid_ref import RowidRef
from .sql_element_cache import SqlElementCache


class FunctionCall(Expr):

    fn: func.Function
    is_method_call: bool
    agg_init_args: dict[str, Any]
    resource_pool: Optional[str]

    # tuple[Optional[int], Optional[Any]]:
    # - for Exprs: (index into components, None)
    # - otherwise: (None, val)
    args: list[tuple[Optional[int], Optional[Any]]]
    kwargs: dict[str, tuple[Optional[int], Optional[Any]]]

    arg_types: list[ts.ColumnType]
    kwarg_types: dict[str, ts.ColumnType]
    group_by_start_idx: int
    group_by_stop_idx: int
    fn_expr_idx: int
    order_by_start_idx: int
    constant_args: set[str]
    aggregator: Optional[Any]
    current_partition_vals: Optional[list[Any]]

    def __init__(
            self, fn: func.Function, bound_args: dict[str, Any], order_by_clause: Optional[list[Any]] = None,
            group_by_clause: Optional[list[Any]] = None, is_method_call: bool = False):
        if order_by_clause is None:
            order_by_clause = []
        if group_by_clause is None:
            group_by_clause = []
        signature = fn.signature
        return_type = fn.call_return_type(bound_args)
        self.fn = fn
        self.is_method_call = is_method_call
        self.normalize_args(fn.name, signature, bound_args)
        self.resource_pool = fn.call_resource_pool(bound_args)

        # If `return_type` is non-nullable, but the function call has a nullable input to any of its non-nullable
        # parameters, then we need to make it nullable. This is because Pixeltable defaults a function output to
        # `None` when any of its non-nullable inputs are `None`.
        for arg_name, arg in bound_args.items():
            param = signature.parameters[arg_name]
            if (
                param.col_type is not None and not param.col_type.nullable
                and isinstance(arg, Expr) and arg.col_type.nullable
            ):
                return_type = return_type.copy(nullable=True)
                break

        super().__init__(return_type)

        self.agg_init_args = {}
        if self.is_agg_fn_call:
            # we separate out the init args for the aggregator
            assert isinstance(fn, func.AggregateFunction)
            self.agg_init_args = {
                arg_name: arg for arg_name, arg in bound_args.items() if arg_name in fn.init_param_names
            }
            bound_args = {arg_name: arg for arg_name, arg in bound_args.items() if arg_name not in fn.init_param_names}

        # construct components, args, kwargs
        self.args = []
        self.kwargs = {}

        # we record the types of non-variable parameters for runtime type checks
        self.arg_types = []
        self.kwarg_types = {}

        # the prefix of parameters that are bound can be passed by position
        processed_args: set[str] = set()
        for py_param in fn.signature.py_signature.parameters.values():
            if py_param.name not in bound_args or py_param.kind == inspect.Parameter.KEYWORD_ONLY:
                break
            arg = bound_args[py_param.name]
            if isinstance(arg, Expr):
                self.args.append((len(self.components), None))
                self.components.append(arg.copy())
            else:
                self.args.append((None, arg))
            if py_param.kind != inspect.Parameter.VAR_POSITIONAL and py_param.kind != inspect.Parameter.VAR_KEYWORD:
                self.arg_types.append(signature.parameters[py_param.name].col_type)
            processed_args.add(py_param.name)

        # the remaining args are passed as keywords
        for param_name in bound_args.keys():
            if param_name not in processed_args:
                arg = bound_args[param_name]
                if isinstance(arg, Expr):
                    self.kwargs[param_name] = (len(self.components), None)
                    self.components.append(arg.copy())
                else:
                    self.kwargs[param_name] = (None, arg)
                if fn.signature.py_signature.parameters[param_name].kind != inspect.Parameter.VAR_KEYWORD:
                    self.kwarg_types[param_name] = signature.parameters[param_name].col_type

        # window function state:
        # self.components[self.group_by_start_idx:self.group_by_stop_idx] contains group_by exprs
        self.group_by_start_idx, self.group_by_stop_idx = 0, 0
        if len(group_by_clause) > 0:
            if isinstance(group_by_clause[0], catalog.Table):
                group_by_exprs = self._create_rowid_refs(group_by_clause[0])
            else:
                assert isinstance(group_by_clause[0], Expr)
                group_by_exprs = group_by_clause
            # record grouping exprs in self.components, we need to evaluate them to get partition vals
            self.group_by_start_idx = len(self.components)
            self.group_by_stop_idx = len(self.components) + len(group_by_exprs)
            self.components.extend(group_by_exprs)

        if isinstance(self.fn, func.ExprTemplateFunction):
            # we instantiate the template to create an Expr that can be evaluated and record that as a component
            fn_expr = self.fn.instantiate(**bound_args)
            self.components.append(fn_expr)
            self.fn_expr_idx = len(self.components) - 1
        else:
            self.fn_expr_idx = sys.maxsize

        # we want to make sure that order_by_clause get assigned slot_idxs, even though we won't need to evaluate them
        # (that's done in SQL)
        if len(order_by_clause) > 0 and not isinstance(order_by_clause[0], Expr):
            raise excs.Error(
                f'order_by argument needs to be a Pixeltable expression, but instead is a {type(order_by_clause[0])}')
        # don't add components after this, everthing after order_by_start_idx is part of the order_by clause
        self.order_by_start_idx = len(self.components)
        self.components.extend(order_by_clause)

        self.constant_args = {param_name for param_name, arg in bound_args.items() if not isinstance(arg, Expr)}
        # execution state for aggregate functions
        self.aggregator = None
        self.current_partition_vals = None

        self.id = self._create_id()

    def _create_rowid_refs(self, tbl: catalog.Table) -> list[Expr]:
        target = tbl._tbl_version_path.tbl_version
        return [RowidRef(target, i) for i in range(target.num_rowid_columns())]

    def default_column_name(self) -> Optional[str]:
        return self.fn.name

    @classmethod
    def normalize_args(cls, fn_name: str, signature: func.Signature, bound_args: dict[str, Any]) -> None:
        """Converts args to Exprs where appropriate and checks that they are compatible with signature.

        Updates bound_args in place, where necessary.
        """
        for param_name, arg in bound_args.items():
            param = signature.parameters[param_name]
            is_var_param = param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)

            if isinstance(arg, dict):
                try:
                    arg = InlineDict(arg)
                    bound_args[param_name] = arg
                    continue
                except excs.Error:
                    # this didn't work, but it might be a literal
                    pass

            if isinstance(arg, list) or isinstance(arg, tuple):
                try:
                    arg = InlineList(arg)
                    bound_args[param_name] = arg
                    continue
                except excs.Error:
                    # this didn't work, but it might be a literal
                    pass

            if not isinstance(arg, Expr):
                # make sure that non-Expr args are json-serializable and are literals of the correct type
                try:
                    _ = json.dumps(arg)
                except TypeError:
                    raise excs.Error(f'Argument for parameter {param_name!r} is not json-serializable: {arg} (of type {type(arg)})')
                if arg is not None:
                    try:
                        param_type = param.col_type
                        bound_args[param_name] = param_type.create_literal(arg)
                    except TypeError as e:
                        msg = str(e)
                        raise excs.Error(f'Argument for parameter {param_name!r}: {msg[0].lower() + msg[1:]}')
                continue

            # these checks break the db migration test, because InlineArray isn't serialized correctly (it looses
            # the type information)
            # if is_var_param:
            #     if param.kind == inspect.Parameter.VAR_POSITIONAL:
            #         if not isinstance(arg, InlineArray) or not arg.col_type.is_json_type():
            #             pass
            #         assert isinstance(arg, InlineArray), type(arg)
            #         assert arg.col_type.is_json_type()
            #     if param.kind == inspect.Parameter.VAR_KEYWORD:
            #         if not isinstance(arg, InlineDict):
            #             pass
            #         assert isinstance(arg, InlineDict), type(arg)
            if is_var_param:
                pass
            else:
                assert param.col_type is not None
                # Check that the argument is consistent with the expected parameter type, with the allowance that
                # non-nullable parameters can still accept nullable arguments (since function calls with Nones
                # assigned to non-nullable parameters will always return None)
                if not (
                    param.col_type.is_supertype_of(arg.col_type, ignore_nullable=True)
                    # TODO: this is a hack to allow JSON columns to be passed to functions that accept scalar
                    # types. It's necessary to avoid littering notebooks with `apply(str)` calls or equivalent.
                    # (Previously, this wasn't necessary because `is_supertype_of()` was improperly implemented.)
                    # We need to think through the right way to handle this scenario.
                    or (arg.col_type.is_json_type() and param.col_type.is_scalar_type())
                ):
                    raise excs.Error(
                        f'Parameter {param_name} (in function {fn_name}): argument type {arg.col_type} does not match parameter type '
                        f'{param.col_type}')

    def _equals(self, other: FunctionCall) -> bool:
        if self.fn != other.fn:
            return False
        if len(self.args) != len(other.args):
            return False
        for i in range(len(self.args)):
            if self.args[i] != other.args[i]:
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
            ('args', self.args),
            ('kwargs', self.kwargs),
            ('group_by_start_idx', self.group_by_start_idx),
            ('group_by_stop_idx', self.group_by_stop_idx),
            ('order_by_start_idx', self.order_by_start_idx)
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
        arg_strs = [
            print_arg(arg) if idx is None else str(self.components[idx]) for idx, arg in self.args[start_idx:]
        ]
        arg_strs.extend([
            f'{param_name}={print_arg(arg) if idx is None else str(self.components[idx])}'
            for param_name, (idx, arg) in self.kwargs.items()
        ])
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
        return self.components[self.group_by_start_idx:self.group_by_stop_idx]

    @property
    def order_by(self) -> list[Expr]:
        return self.components[self.order_by_start_idx:]

    @property
    def is_window_fn_call(self) -> bool:
        return isinstance(self.fn, func.AggregateFunction) and self.fn.allows_window and (
            not self.fn.allows_std_agg
            or self.has_group_by()
            or (len(self.order_by) > 0 and not self.fn.requires_order_by)
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
        for param_name, (component_idx, arg) in self.kwargs.items():
            param = self.fn.signature.parameters[param_name]
            assert param.kind != inspect.Parameter.VAR_POSITIONAL and param.kind != inspect.Parameter.VAR_KEYWORD
            if component_idx is None:
                kwargs[param_name] = sql.literal(arg)
            else:
                arg_element = sql_elements.get(self.components[component_idx])
                if arg_element is None:
                    return None
                kwargs[param_name] = arg_element

        args: list[sql.ColumnElement] = []
        for _, (component_idx, arg) in enumerate(self.args):
            if component_idx is None:
                args.append(sql.literal(arg))
            else:
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
        self.aggregator = self.fn.agg_cls(**self.agg_init_args)

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
        for param_name, (component_idx, arg) in self.kwargs.items():
            val = arg if component_idx is None else data_row[self.components[component_idx].slot_idx]
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
        for param_idx, (component_idx, arg) in enumerate(self.args):
            val = arg if component_idx is None else data_row[self.components[component_idx].slot_idx]
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
            if self.has_group_by():
                if self.current_partition_vals is None:
                    self.current_partition_vals = [None] * len(self.group_by)
                partition_vals = [data_row[e.slot_idx] for e in self.group_by]
                if partition_vals != self.current_partition_vals:
                    # new partition
                    self.aggregator = self.fn.agg_cls(**self.agg_init_args)
                    self.current_partition_vals = partition_vals
            elif self.aggregator is None:
                self.aggregator = self.fn.agg_cls(**self.agg_init_args)
            self.aggregator.update(*args)
            data_row[self.slot_idx] = self.aggregator.value()
        else:
            data_row[self.slot_idx] = self.fn.exec(*args, **kwargs)

    def _as_dict(self) -> dict:
        result = {
            'fn': self.fn.as_dict(), 'args': self.args, 'kwargs': self.kwargs,
            'group_by_start_idx': self.group_by_start_idx, 'group_by_stop_idx': self.group_by_stop_idx,
            'order_by_start_idx': self.order_by_start_idx,
            **super()._as_dict()
        }
        return result

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> FunctionCall:
        assert 'fn' in d
        assert 'args' in d
        assert 'kwargs' in d
        # reassemble bound args
        fn = func.Function.from_dict(d['fn'])
        param_names = list(fn.signature.parameters.keys())
        bound_args = {param_names[i]: arg if idx is None else components[idx] for i, (idx, arg) in enumerate(d['args'])}
        bound_args.update(
            {param_name: val if idx is None else components[idx] for param_name, (idx, val) in d['kwargs'].items()})
        group_by_exprs = components[d['group_by_start_idx']:d['group_by_stop_idx']]
        order_by_exprs = components[d['order_by_start_idx']:]
        fn_call = cls(
            func.Function.from_dict(d['fn']), bound_args, group_by_clause=group_by_exprs,
            order_by_clause=order_by_exprs)
        return fn_call

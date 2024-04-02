from __future__ import annotations

import inspect
import json
import sys
from typing import Optional, List, Any, Dict, Tuple

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exceptions as excs
import pixeltable.func as func
import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .inline_array import InlineArray
from .inline_dict import InlineDict
from .row_builder import RowBuilder
from .rowid_ref import RowidRef


class FunctionCall(Expr):
    def __init__(
            self, fn: func.Function, bound_args: Dict[str, Any], order_by_clause: Optional[List[Any]] = None,
            group_by_clause: Optional[List[Any]] = None, is_method_call: bool = False):
        if order_by_clause is None:
            order_by_clause = []
        if group_by_clause is None:
            group_by_clause = []
        signature = fn.signature
        super().__init__(signature.get_return_type(bound_args))
        self.fn = fn
        self.is_method_call = is_method_call
        self.check_args(signature, bound_args)

        self.agg_init_args: Dict[str, Any] = {}
        if self.is_agg_fn_call:
            # we separate out the init args for the aggregator
            self.agg_init_args = {
                arg_name: arg for arg_name, arg in bound_args.items() if arg_name in fn.init_param_names
            }
            bound_args = {arg_name: arg for arg_name, arg in bound_args.items() if arg_name not in fn.init_param_names}

        # construct components, args, kwargs
        self.components: List[Expr] = []

        # Tuple[int, Any]:
        # - for Exprs: (index into components, None)
        # - otherwise: (-1, val)
        self.args: List[Tuple[int, Any]] = []
        self.kwargs: Dict[str, Tuple[int, Any]] = {}

        # we record the types of non-variable parameters for runtime type checks
        self.arg_types: List[ts.ColumnType] = []
        self.kwarg_types: Dict[str, ts.ColumnType] = {}
        # the prefix of parameters that are bound can be passed by position
        for param in fn.py_signature.parameters.values():
            if param.name not in bound_args or param.kind == inspect.Parameter.KEYWORD_ONLY:
                break
            arg = bound_args[param.name]
            if isinstance(arg, Expr):
                self.args.append((len(self.components), None))
                self.components.append(arg.copy())
            else:
                self.args.append((-1, arg))
            if param.kind != inspect.Parameter.VAR_POSITIONAL and param.kind != inspect.Parameter.VAR_KEYWORD:
                self.arg_types.append(signature.parameters[param.name].col_type)

        # the remaining args are passed as keywords
        kw_param_names = set(bound_args.keys()) - set(list(fn.py_signature.parameters.keys())[:len(self.args)])
        for param_name in kw_param_names:
            arg = bound_args[param_name]
            if isinstance(arg, Expr):
                self.kwargs[param_name] = (len(self.components), None)
                self.components.append(arg.copy())
            else:
                self.kwargs[param_name] = (-1, arg)
            if fn.py_signature.parameters[param_name].kind != inspect.Parameter.VAR_KEYWORD:
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
        self.aggregator: Optional[Any] = None
        self.current_partition_vals: Optional[List[Any]] = None

        self.id = self._create_id()

    def _create_rowid_refs(self, tbl: catalog.Table) -> List[Expr]:
        target = tbl.tbl_version_path.tbl_version
        return [RowidRef(target, i) for i in range(target.num_rowid_columns())]

    @classmethod
    def check_args(cls, signature: func.Signature, bound_args: Dict[str, Any]) -> None:
        """Checks that bound_args are compatible with signature.

        Convert literals to the correct type and update bound_args in place, if necessary.
        """
        for param_name, arg in bound_args.items():
            param = signature.parameters[param_name]
            if isinstance(arg, dict):
                try:
                    arg = InlineDict(arg)
                    bound_args[param_name] = arg
                except excs.Error:
                    # this didn't work, but it might be a literal
                    pass
            if isinstance(arg, list) or isinstance(arg, tuple):
                try:
                    # If the column type is JsonType, force the literal to be JSON
                    arg = InlineArray(arg, force_json=param.col_type is not None and param.col_type.is_json_type())
                    bound_args[param_name] = arg
                except excs.Error:
                    # this didn't work, but it might be a literal
                    pass

            if not isinstance(arg, Expr):
                # make sure that non-Expr args are json-serializable and are literals of the correct type
                try:
                    _ = json.dumps(arg)
                except TypeError:
                    raise excs.Error(f"Argument for parameter '{param_name}' is not json-serializable: {arg}")
                if arg is not None:
                    try:
                        param_type = param.col_type
                        bound_args[param_name] = param_type.create_literal(arg)
                    except TypeError as e:
                        msg = str(e)
                        raise excs.Error(f"Argument for parameter '{param_name}': {msg[0].lower() + msg[1:]}")
                continue

            # variable parameters don't get type-checked, but they both need to be json-typed
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                assert isinstance(arg, InlineArray)
                arg.col_type = ts.JsonType()
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                assert isinstance(arg, InlineDict)
                arg.col_type = ts.JsonType()
            continue

            if not param_type.is_supertype_of(arg.col_type):
                raise excs.Error(
                    f'Parameter {param_name}: argument type {arg.col_type} does not match parameter type '
                    f'{param_type}')

    def is_nos_call(self) -> bool:
        return isinstance(self.fn, func.NOSFunction)

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

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [
            ('fn', id(self.fn)),  # use the function pointer, not the fqn, which isn't set for lambdas
            ('args', self.args),
            ('kwargs', self.kwargs),
            ('group_by_start_idx', self.group_by_start_idx),
            ('group_by_stop_idx', self.group_by_stop_idx),
            ('order_by_start_idx', self.order_by_start_idx)
        ]

    def __str__(self) -> str:
        return self.display_str()

    def display_str(self, inline: bool = True) -> str:
        if self.is_method_call:
            return f'{self.components[0]}.{self.fn.name}({self._print_args(1, inline)})'
        else:
            fn_name = self.fn.display_name if self.fn.display_name != '' else 'anonymous_fn'
            return f'{fn_name}({self._print_args()})'

    def _print_args(self, start_idx: int = 0, inline: bool = True) -> str:
        arg_strs = [
            str(arg) if idx == -1 else str(self.components[idx]) for idx, arg in self.args[start_idx:]
        ]
        def print_arg(arg: Any) -> str:
            return f"'{arg}'" if isinstance(arg, str) else str(arg)
        arg_strs.extend([
            f'{param_name}={print_arg(arg) if idx == -1 else str(self.components[idx])}'
            for param_name, (idx, arg) in self.kwargs.items()
        ])
        if len(self.order_by) > 0:
            if self.fn.requires_order_by:
                arg_strs.insert(0, Expr.print_list(self.order_by))
            else:
                arg_strs.append(f'order_by={Expr.print_list(self.order_by)}')
        if len(self.group_by) > 0:
            arg_strs.append(f'group_by={Expr.print_list(self.group_by)}')
        # TODO: figure out the function name
        separator = ', ' if inline else ',\n    '
        return separator.join(arg_strs)

    def has_group_by(self) -> List[Expr]:
        return self.group_by_stop_idx != 0

    @property
    def group_by(self) -> List[Expr]:
        return self.components[self.group_by_start_idx:self.group_by_stop_idx]

    @property
    def order_by(self) -> List[Expr]:
        return self.components[self.order_by_start_idx:]

    @property
    def is_window_fn_call(self) -> bool:
        return isinstance(self.fn, func.AggregateFunction) and self.fn.allows_window and \
            (not self.fn.allows_std_agg \
             or self.has_group_by() \
             or (len(self.order_by) > 0 and not self.fn.requires_order_by))

    def get_window_sort_exprs(self) -> Tuple[List[Expr], List[Expr]]:
        return self.group_by, self.order_by

    @property
    def is_agg_fn_call(self) -> bool:
        return isinstance(self.fn, func.AggregateFunction)

    def get_agg_order_by(self) -> List[Expr]:
        assert self.is_agg_fn_call
        return self.order_by

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        # TODO: implement for standard aggregate functions
        return None

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
        args, kwargs = self._make_args(data_row)
        self.aggregator.update(*args, **kwargs)

    def _make_args(self, data_row: DataRow) -> Tuple[List[Any], Dict[str, Any]]:
        """Return args and kwargs, constructed for data_row"""
        kwargs: Dict[str, Any] = {}
        for param_name, (component_idx, arg) in self.kwargs.items():
            val = arg if component_idx == -1 else data_row[self.components[component_idx].slot_idx]
            param = self.fn.signature.parameters[param_name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                # expand **kwargs parameter
                kwargs.update(val)
            else:
                assert param.kind != inspect.Parameter.VAR_POSITIONAL
                kwargs[param_name] = val

        args: List[Any] = []
        for param_idx, (component_idx, arg) in enumerate(self.args):
            val = arg if component_idx == -1 else data_row[self.components[component_idx].slot_idx]
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
                args.append(val)
        return args, kwargs

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        args, kwargs = self._make_args(data_row)
        signature = self.fn.signature
        if signature.parameters is not None:
            # check for nulls
            for i in range(len(self.arg_types)):
                if args[i] is None and not self.arg_types[i].nullable:
                    # we can't evaluate this function
                    data_row[self.slot_idx] = None
                    return
            for param_name, param_type in self.kwarg_types.items():
                if kwargs[param_name] is None and not param_type.nullable:
                    # we can't evaluate this function
                    data_row[self.slot_idx] = None
                    return

        if isinstance(self.fn, func.ExprTemplateFunction):
            # we need to evaluate the template
            # TODO: can we get rid of this extra copy?
            fn_expr = self.components[self.fn_expr_idx]
            data_row[self.slot_idx] = data_row[fn_expr.slot_idx]
        elif isinstance(self.fn, func.CallableFunction):
            data_row[self.slot_idx] = self.fn.py_fn(*args, **kwargs)
        elif self.is_window_fn_call:
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
            assert self.is_agg_fn_call
            data_row[self.slot_idx] = self.aggregator.value()

    def _as_dict(self) -> Dict:
        result = {
            'fn': self.fn.as_dict(), 'args': self.args, 'kwargs': self.kwargs,
            'group_by_start_idx': self.group_by_start_idx, 'group_by_stop_idx': self.group_by_stop_idx,
            'order_by_start_idx': self.order_by_start_idx,
            **super()._as_dict()
        }
        return result

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'fn' in d
        assert 'args' in d
        assert 'kwargs' in d
        # reassemble bound args
        fn = func.Function.from_dict(d['fn'])
        param_names = list(fn.signature.parameters.keys())
        bound_args = {param_names[i]: arg if idx == -1 else components[idx] for i, (idx, arg) in enumerate(d['args'])}
        bound_args.update(
            {param_name: val if idx == -1 else components[idx] for param_name, (idx, val) in d['kwargs'].items()})
        group_by_exprs = components[d['group_by_start_idx']:d['group_by_stop_idx']]
        order_by_exprs = components[d['order_by_start_idx']:]
        fn_call = cls(
            func.Function.from_dict(d['fn']), bound_args, group_by_clause=group_by_exprs,
            order_by_clause=order_by_exprs)
        return fn_call

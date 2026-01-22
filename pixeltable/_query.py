from __future__ import annotations

import builtins
import copy
import dataclasses
import hashlib
import json
import logging
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Hashable, Iterator, NoReturn, Sequence, TypeVar

import pandas as pd
import pydantic
import sqlalchemy.exc as sql_exc

from pixeltable import catalog, exceptions as excs, exec, exprs, plan, type_system as ts
from pixeltable.catalog import Catalog, is_valid_identifier
from pixeltable.catalog.update_status import UpdateStatus
from pixeltable.env import Env
from pixeltable.plan import Planner, SampleClause
from pixeltable.type_system import ColumnType
from pixeltable.utils.description_helper import DescriptionHelper
from pixeltable.utils.formatter import Formatter

if TYPE_CHECKING:
    import torch
    import torch.utils.data

__all__ = ['Query']

_logger = logging.getLogger('pixeltable')


class ResultSet:
    _rows: list[list[Any]]
    _col_names: list[str]
    __schema: dict[str, ColumnType]
    __formatter: Formatter

    def __init__(self, rows: list[list[Any]], schema: dict[str, ColumnType]):
        self._rows = rows
        self._col_names = list(schema.keys())
        self.__schema = schema
        self.__formatter = Formatter(len(self._rows), len(self._col_names), Env.get().http_address)

    @property
    def schema(self) -> dict[str, ColumnType]:
        return self.__schema

    def __len__(self) -> int:
        return len(self._rows)

    def __repr__(self) -> str:
        return self.to_pandas().__repr__()

    def _repr_html_(self) -> str:
        formatters: dict[Hashable, Callable[[object], str]] = {}
        for col_name, col_type in self.schema.items():
            formatter = self.__formatter.get_pandas_formatter(col_type)
            if formatter is not None:
                formatters[col_name] = formatter
        return self.to_pandas().to_html(formatters=formatters, escape=False, index=False)

    def __str__(self) -> str:
        return self.to_pandas().to_string()

    def _reverse(self) -> None:
        """Reverse order of rows"""
        self._rows.reverse()

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._rows, columns=self._col_names)

    BaseModelT = TypeVar('BaseModelT', bound=pydantic.BaseModel)

    def to_pydantic(self, model: type[BaseModelT]) -> Iterator[BaseModelT]:
        """
        Convert the ResultSet to a list of Pydantic model instances.

        Args:
            model: A Pydantic model class.

        Returns:
            An iterator over Pydantic model instances, one for each row in the result set.

        Raises:
            Error: If the row data doesn't match the model schema.
        """
        model_fields = model.model_fields
        model_config = getattr(model, 'model_config', {})
        forbid_extra_fields = model_config.get('extra') == 'forbid'

        # schema validation
        required_fields = {name for name, field in model_fields.items() if field.is_required()}
        col_names = set(self._col_names)
        missing_fields = required_fields - col_names
        if len(missing_fields) > 0:
            raise excs.Error(
                f'Required model fields {missing_fields} are missing from result set columns {self._col_names}'
            )
        if forbid_extra_fields:
            extra_fields = col_names - set(model_fields.keys())
            if len(extra_fields) > 0:
                raise excs.Error(f"Extra fields {extra_fields} are not allowed in model with extra='forbid'")

        for row in self:
            try:
                yield model(**row)
            except pydantic.ValidationError as e:
                raise excs.Error(str(e)) from e

    def _row_to_dict(self, row_idx: int) -> dict[str, Any]:
        return {self._col_names[i]: self._rows[row_idx][i] for i in range(len(self._col_names))}

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, str):
            if index not in self._col_names:
                raise excs.Error(f'Invalid column name: {index}')
            col_idx = self._col_names.index(index)
            return [row[col_idx] for row in self._rows]
        if isinstance(index, int):
            return self._row_to_dict(index)
        if isinstance(index, tuple) and len(index) == 2:
            if not isinstance(index[0], int) or not isinstance(index[1], (str, int)):
                raise excs.Error(f'Bad index, expected [<row idx>, <column name | column index>]: {index}')
            if isinstance(index[1], str) and index[1] not in self._col_names:
                raise excs.Error(f'Invalid column name: {index[1]}')
            col_idx = self._col_names.index(index[1]) if isinstance(index[1], str) else index[1]
            return self._rows[index[0]][col_idx]
        raise excs.Error(f'Bad index: {index}')

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (self._row_to_dict(i) for i in range(len(self)))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResultSet):
            return False
        return self.to_pandas().equals(other.to_pandas())

    def __hash__(self) -> int:
        return hash(self.to_pandas())


# # TODO: remove this; it's only here as a reminder that we still need to call release() in the current implementation
# class AnalysisInfo:
#     def __init__(self, tbl: catalog.TableVersion):
#         self.tbl = tbl
#         # output of the SQL scan stage
#         self.sql_scan_output_exprs: list[exprs.Expr] = []
#         # output of the agg stage
#         self.agg_output_exprs: list[exprs.Expr] = []
#         # Where clause of the Select stmt of the SQL scan stage
#         self.sql_where_clause: sql.ClauseElement | None = None
#         # filter predicate applied to input rows of the SQL scan stage
#         self.filter: exprs.Predicate | None = None
#         self.similarity_clause: exprs.ImageSimilarityPredicate | None = None
#         self.agg_fn_calls: list[exprs.FunctionCall] = []  # derived from unique_exprs
#         self.has_frame_col: bool = False  # True if we're referencing the frame col
#
#         self.evaluator: exprs.Evaluator | None = None
#         self.sql_scan_eval_ctx: list[exprs.Expr] = []  # needed to materialize output of SQL scan stage
#         self.agg_eval_ctx: list[exprs.Expr] = []  # needed to materialize output of agg stage
#         self.filter_eval_ctx: list[exprs.Expr] = []
#         self.group_by_eval_ctx: list[exprs.Expr] = []
#
#     def finalize_exec(self) -> None:
#         """
#         Call release() on all collected Exprs.
#         """
#         exprs.Expr.release_list(self.sql_scan_output_exprs)
#         exprs.Expr.release_list(self.agg_output_exprs)
#         if self.filter is not None:
#             self.filter.release()


class Query:
    """Represents a query for retrieving and transforming data from Pixeltable tables."""

    _from_clause: plan.FromClause
    _select_list_exprs: list[exprs.Expr]
    _schema: dict[str, ts.ColumnType]
    select_list: list[tuple[exprs.Expr, str | None]] | None
    where_clause: exprs.Expr | None
    group_by_clause: list[exprs.Expr] | None
    grouping_tbl: catalog.TableVersion | None
    order_by_clause: list[tuple[exprs.Expr, bool]] | None
    limit_val: exprs.Expr | None
    sample_clause: SampleClause | None

    def __init__(
        self,
        from_clause: plan.FromClause | None = None,
        select_list: list[tuple[exprs.Expr, str | None]] | None = None,
        where_clause: exprs.Expr | None = None,
        group_by_clause: list[exprs.Expr] | None = None,
        grouping_tbl: catalog.TableVersion | None = None,
        order_by_clause: list[tuple[exprs.Expr, bool]] | None = None,  # list[(expr, asc)]
        limit: exprs.Expr | None = None,
        sample_clause: SampleClause | None = None,
    ):
        self._from_clause = from_clause

        # exprs contain execution state and therefore cannot be shared
        select_list = copy.deepcopy(select_list)
        select_list_exprs, column_names = Query._normalize_select_list(self._from_clause.tbls, select_list)
        # check select list after expansion to catch early
        # the following two lists are always non empty, even if select list is None.
        assert len(column_names) == len(select_list_exprs)
        self._select_list_exprs = select_list_exprs
        self._schema = {column_names[i]: select_list_exprs[i].col_type for i in range(len(column_names))}
        self.select_list = select_list

        self.where_clause = copy.deepcopy(where_clause)
        assert group_by_clause is None or grouping_tbl is None
        self.group_by_clause = copy.deepcopy(group_by_clause)
        self.grouping_tbl = grouping_tbl
        self.order_by_clause = copy.deepcopy(order_by_clause)
        self.limit_val = limit
        self.sample_clause = sample_clause

    @classmethod
    def _normalize_select_list(
        cls, tbls: list[catalog.TableVersionPath], select_list: list[tuple[exprs.Expr, str | None]] | None
    ) -> tuple[list[exprs.Expr], list[str]]:
        """
        Expand select list information with all columns and their names
        Returns:
            a pair composed of the list of expressions and the list of corresponding names
        """
        if select_list is None:
            select_list = [(exprs.ColumnRef(col), None) for tbl in tbls for col in tbl.columns()]

        out_exprs: list[exprs.Expr] = []
        out_names: list[str] = []  # keep track of order
        seen_out_names: set[str] = set()  # use to check for duplicates in loop, avoid square complexity
        for i, (expr, name) in enumerate(select_list):
            if name is None:
                # use default, add suffix if needed so default adds no duplicates
                default_name = expr.default_column_name()
                if default_name is not None:
                    column_name = default_name
                    if default_name in seen_out_names:
                        # already used, then add suffix until unique name is found
                        for j in range(1, len(out_names) + 1):
                            column_name = f'{default_name}_{j}'
                            if column_name not in seen_out_names:
                                break
                else:  # no default name, eg some expressions
                    column_name = f'col_{i}'
            else:  # user provided name, no attempt to rename
                column_name = name

            out_exprs.append(expr)
            out_names.append(column_name)
            seen_out_names.add(column_name)
        assert len(out_exprs) == len(out_names)
        assert set(out_names) == seen_out_names
        return out_exprs, out_names

    @property
    def _first_tbl(self) -> catalog.TableVersionPath:
        return self._from_clause._first_tbl

    def _vars(self) -> dict[str, exprs.Variable]:
        """
        Return a dict mapping variable name to Variable for all Variables contained in any component of the Query
        """
        all_exprs: list[exprs.Expr] = []
        all_exprs.extend(self._select_list_exprs)
        if self.where_clause is not None:
            all_exprs.append(self.where_clause)
        if self.group_by_clause is not None:
            all_exprs.extend(self.group_by_clause)
        if self.order_by_clause is not None:
            all_exprs.extend([expr for expr, _ in self.order_by_clause])
        if self.limit_val is not None:
            all_exprs.append(self.limit_val)
        vars = exprs.Expr.list_subexprs(all_exprs, expr_class=exprs.Variable)
        unique_vars: dict[str, exprs.Variable] = {}
        for var in vars:
            if var.name not in unique_vars:
                unique_vars[var.name] = var
            elif unique_vars[var.name].col_type != var.col_type:
                raise excs.Error(f'Multiple definitions of parameter {var.name!r}')
        return unique_vars

    @classmethod
    def _convert_param_to_typed_expr(
        cls, v: Any, required_type: ts.ColumnType, required: bool, name: str, range: tuple[Any, Any] | None = None
    ) -> exprs.Expr | None:
        if v is None:
            if required:
                raise excs.Error(f'{name!r} parameter must be present')
            return v
        v_expr = exprs.Expr.from_object(v)
        if not v_expr.col_type.matches(required_type):
            raise excs.Error(f'{name!r} parameter must be of type `{required_type}`; got `{v_expr.col_type}`')
        if range is not None:
            if not isinstance(v_expr, exprs.Literal):
                raise excs.Error(f'{name!r} parameter must be a constant; got: {v_expr}')
            if range[0] is not None and not (v_expr.val >= range[0]):
                raise excs.Error(f'{name!r} parameter must be >= {range[0]}')
            if range[1] is not None and not (v_expr.val <= range[1]):
                raise excs.Error(f'{name!r} parameter must be <= {range[1]}')
        return v_expr

    @classmethod
    def validate_constant_type_range(
        cls, v: Any, required_type: ts.ColumnType, required: bool, name: str, range: tuple[Any, Any] | None = None
    ) -> Any:
        """Validate that the given named parameter is a constant of the required type and within the specified range."""
        v_expr = cls._convert_param_to_typed_expr(v, required_type, required, name, range)
        if v_expr is None:
            return None
        return v_expr.val

    def parameters(self) -> dict[str, ColumnType]:
        """Return a dict mapping parameter name to parameter type.

        Parameters are Variables contained in any component of the Query.
        """
        return {name: var.col_type for name, var in self._vars().items()}

    def _exec(self) -> Iterator[exprs.DataRow]:
        """Run the query and return rows as a generator.
        This function must not modify the state of the Query, otherwise it breaks dataset caching.
        """
        plan = self._create_query_plan()

        def exec_plan() -> Iterator[exprs.DataRow]:
            with plan:
                for row_batch in plan:
                    # stop progress output before we display anything, otherwise it'll mess up the output
                    Env.get().stop_progress()
                    yield from row_batch

        yield from exec_plan()

    async def _aexec(self) -> AsyncIterator[exprs.DataRow]:
        """Run the query and return rows as a generator.
        This function must not modify the state of the Query, otherwise it breaks dataset caching.
        """
        plan = self._create_query_plan()
        with plan:
            async for row_batch in plan:
                for row in row_batch:
                    yield row

    def _create_query_plan(self) -> exec.ExecNode:
        # construct a group-by clause if we're grouping by a table
        group_by_clause: list[exprs.Expr] | None = None
        if self.grouping_tbl is not None:
            assert self.group_by_clause is None
            num_rowid_cols = len(self.grouping_tbl.store_tbl.rowid_columns())
            # the grouping table must be a base of self.tbl
            assert num_rowid_cols <= len(self._first_tbl.tbl_version.get().store_tbl.rowid_columns())
            group_by_clause = self.__rowid_columns(num_rowid_cols)
        elif self.group_by_clause is not None:
            group_by_clause = self.group_by_clause

        for item in self._select_list_exprs:
            item.bind_rel_paths()

        return Planner.create_query_plan(
            self._from_clause,
            self._select_list_exprs,
            where_clause=self.where_clause,
            group_by_clause=group_by_clause,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            sample_clause=self.sample_clause,
        )

    def __rowid_columns(self, num_rowid_cols: int | None = None) -> list[exprs.Expr]:
        """Return list of RowidRef for the given number of associated rowids"""
        return Planner.rowid_columns(self._first_tbl.tbl_version, num_rowid_cols)

    def _has_joins(self) -> bool:
        return len(self._from_clause.join_clauses) > 0

    def show(self, n: int = 20) -> ResultSet:
        if self.sample_clause is not None:
            raise excs.Error('show() cannot be used with sample()')
        assert n is not None
        return self.limit(n).collect()

    def head(self, n: int = 10) -> ResultSet:
        """Return the first n rows of the Query, in insertion order of the underlying Table.

        head() is not supported for joins.

        Args:
            n: Number of rows to select. Default is 10.

        Returns:
            A ResultSet with the first n rows of the Query.

        Raises:
            Error: If the Query is the result of a join or
                if the Query has an order_by clause.
        """
        if self.order_by_clause is not None:
            raise excs.Error('head() cannot be used with order_by()')
        if self._has_joins():
            raise excs.Error('head() not supported for joins')
        if self.sample_clause is not None:
            raise excs.Error('head() cannot be used with sample()')
        if self.group_by_clause is not None:
            raise excs.Error('head() cannot be used with group_by()')
        num_rowid_cols = len(self._first_tbl.tbl_version.get().store_tbl.rowid_columns())
        order_by_clause = [exprs.RowidRef(self._first_tbl.tbl_version, idx) for idx in range(num_rowid_cols)]
        return self.order_by(*order_by_clause, asc=True).limit(n).collect()

    def tail(self, n: int = 10) -> ResultSet:
        """Return the last n rows of the Query, in insertion order of the underlying Table.

        tail() is not supported for joins.

        Args:
            n: Number of rows to select. Default is 10.

        Returns:
            A ResultSet with the last n rows of the Query.

        Raises:
            Error: If the Query is the result of a join or
                if the Query has an order_by clause.
        """
        if self.order_by_clause is not None:
            raise excs.Error('tail() cannot be used with order_by()')
        if self._has_joins():
            raise excs.Error('tail() not supported for joins')
        if self.sample_clause is not None:
            raise excs.Error('tail() cannot be used with sample()')
        if self.group_by_clause is not None:
            raise excs.Error('tail() cannot be used with group_by()')
        num_rowid_cols = len(self._first_tbl.tbl_version.get().store_tbl.rowid_columns())
        order_by_clause = [exprs.RowidRef(self._first_tbl.tbl_version, idx) for idx in range(num_rowid_cols)]
        result = self.order_by(*order_by_clause, asc=False).limit(n).collect()
        result._reverse()
        return result

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Column names and types in this Query."""
        return self._schema

    def bind(self, args: dict[str, Any]) -> Query:
        """Bind arguments to parameters and return a new Query."""
        # substitute Variables with the corresponding values according to 'args', converted to Literals
        select_list_exprs = copy.deepcopy(self._select_list_exprs)
        where_clause = copy.deepcopy(self.where_clause)
        group_by_clause = copy.deepcopy(self.group_by_clause)
        order_by_exprs = (
            [copy.deepcopy(order_by_expr) for order_by_expr, _ in self.order_by_clause]
            if self.order_by_clause is not None
            else None
        )
        limit_val = copy.deepcopy(self.limit_val)

        var_exprs: dict[exprs.Expr, exprs.Expr] = {}
        vars = self._vars()
        for arg_name, arg_val in args.items():
            if arg_name not in vars:
                # ignore unused variables
                continue
            var_expr = vars[arg_name]
            arg_expr = exprs.Expr.from_object(arg_val)
            if arg_expr is None:
                raise excs.Error(f'That argument cannot be converted to a Pixeltable expression: {arg_val}')
            var_exprs[var_expr] = arg_expr

        exprs.Expr.list_substitute(select_list_exprs, var_exprs)
        if where_clause is not None:
            where_clause = where_clause.substitute(var_exprs)
        if group_by_clause is not None:
            exprs.Expr.list_substitute(group_by_clause, var_exprs)
        if order_by_exprs is not None:
            exprs.Expr.list_substitute(order_by_exprs, var_exprs)

        select_list = list(zip(select_list_exprs, self.schema.keys()))
        order_by_clause: list[tuple[exprs.Expr, bool]] | None = None
        if order_by_exprs is not None:
            order_by_clause = [
                (expr, asc) for expr, asc in zip(order_by_exprs, [asc for _, asc in self.order_by_clause])
            ]
        if limit_val is not None:
            limit_val = limit_val.substitute(var_exprs)
            if limit_val is not None and not isinstance(limit_val, exprs.Literal):
                raise excs.Error(f'limit(): parameter must be a constant; got: {limit_val}')

        return Query(
            from_clause=self._from_clause,
            select_list=select_list,
            where_clause=where_clause,
            group_by_clause=group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=order_by_clause,
            limit=limit_val,
        )

    def _raise_expr_eval_err(self, e: excs.ExprEvalError) -> NoReturn:
        msg = f'In row {e.row_num} the {e.expr_msg} encountered exception {type(e.exc).__name__}:\n{e.exc}'
        if len(e.input_vals) > 0:
            input_msgs = [
                f"'{d}' = {d.col_type.print_value(e.input_vals[i])}" for i, d in enumerate(e.expr.dependencies())
            ]
            msg += f'\nwith {", ".join(input_msgs)}'
        assert e.exc_tb is not None
        stack_trace = traceback.format_tb(e.exc_tb)
        if len(stack_trace) > 2:
            # append a stack trace if the exception happened in user code
            # (frame 0 is ExprEvaluator and frame 1 is some expr's eval()
            nl = '\n'
            # [-1:0:-1]: leave out entry 0 and reverse order, so that the most recent frame is at the top
            msg += f'\nStack:\n{nl.join(stack_trace[-1:1:-1])}'
        raise excs.Error(msg) from e

    def _output_row_iterator(self) -> Iterator[list]:
        # TODO: extend begin_xact() to accept multiple TVPs for joins
        single_tbl = self._first_tbl if len(self._from_clause.tbls) == 1 else None
        with Catalog.get().begin_xact(tbl=single_tbl, for_write=False):
            try:
                for data_row in self._exec():
                    yield [data_row[e.slot_idx] for e in self._select_list_exprs]
            except excs.ExprEvalError as e:
                self._raise_expr_eval_err(e)
            except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
                Catalog.get().convert_sql_exc(e, tbl=(single_tbl.tbl_version if single_tbl is not None else None))
                raise  # just re-raise if not converted to a Pixeltable error

    def collect(self) -> ResultSet:
        return ResultSet(list(self._output_row_iterator()), self.schema)

    async def _acollect(self) -> ResultSet:
        single_tbl = self._first_tbl if len(self._from_clause.tbls) == 1 else None
        try:
            result = [[row[e.slot_idx] for e in self._select_list_exprs] async for row in self._aexec()]
            return ResultSet(result, self.schema)
        except excs.ExprEvalError as e:
            self._raise_expr_eval_err(e)
        except (sql_exc.DBAPIError, sql_exc.OperationalError, sql_exc.InternalError) as e:
            Catalog.get().convert_sql_exc(e, tbl=(single_tbl.tbl_version if single_tbl is not None else None))
            raise  # just re-raise if not converted to a Pixeltable error

    def count(self) -> int:
        """Return the number of rows in the Query.

        Returns:
            The number of rows in the Query.
        """
        with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=False) as conn:
            count_stmt = Planner.create_count_stmt(self)
            result: int = conn.execute(count_stmt).scalar_one()
            assert isinstance(result, int)
            return result

    def _descriptors(self) -> DescriptionHelper:
        helper = DescriptionHelper()
        helper.append(self._col_descriptor())
        qd = self._query_descriptor()
        if not qd.empty:
            helper.append(qd, show_index=True, show_header=False)
        return helper

    def _col_descriptor(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    'Name': name,
                    'Type': expr.col_type._to_str(as_schema=True),
                    'Expression': expr.display_str(inline=False),
                }
                for name, expr in zip(self.schema.keys(), self._select_list_exprs)
            ]
        )

    def _query_descriptor(self) -> pd.DataFrame:
        heading_vals: list[str] = []
        info_vals: list[str] = []
        heading_vals.append('From')
        info_vals.extend(tbl.tbl_name() for tbl in self._from_clause.tbls)
        if self.where_clause is not None:
            heading_vals.append('Where')
            info_vals.append(self.where_clause.display_str(inline=False))
        if self.group_by_clause is not None:
            heading_vals.append('Group By')
            heading_vals.extend([''] * (len(self.group_by_clause) - 1))
            info_vals.extend(e.display_str(inline=False) for e in self.group_by_clause)
        if self.order_by_clause is not None:
            heading_vals.append('Order By')
            heading_vals.extend([''] * (len(self.order_by_clause) - 1))
            info_vals.extend(
                [f'{e[0].display_str(inline=False)} {"asc" if e[1] else "desc"}' for e in self.order_by_clause]
            )
        if self.limit_val is not None:
            heading_vals.append('Limit')
            info_vals.append(self.limit_val.display_str(inline=False))
        if self.sample_clause is not None:
            heading_vals.append('Sample')
            info_vals.append(self.sample_clause.display_str(inline=False))
        assert len(heading_vals) == len(info_vals)
        return pd.DataFrame(info_vals, index=heading_vals)

    def describe(self) -> None:
        """
        Prints a tabular description of this Query.
        The description has two columns, heading and info, which list the contents of each 'component'
                (select list, where clause, ...) vertically.
        """
        if getattr(builtins, '__IPYTHON__', False):
            from IPython.display import Markdown, display

            display(Markdown(self._repr_html_()))
        else:
            print(repr(self))

    def __repr__(self) -> str:
        return self._descriptors().to_string()

    def _repr_html_(self) -> str:
        return self._descriptors().to_html()

    def select(self, *items: Any, **named_items: Any) -> Query:
        """Select columns or expressions from the Query.

        Args:
            items: expressions to be selected
            named_items: named expressions to be selected

        Returns:
            A new Query with the specified select list.

        Raises:
            Error: If the select list is already specified,
                or if any of the specified expressions are invalid,
                or refer to tables not in the Query.

        Examples:
            Given the Query person from a table t with all its columns and rows:

            >>> person = t.select()

            Select the columns 'name' and 'age' (referenced in table t) from the Query person:

            >>> query = person.select(t.name, t.age)

            Select the columns 'name' (referenced in table t) from the Query person,
            and a named column 'is_adult' from the expression `age >= 18` where 'age' is
            another column in table t:

            >>> query = person.select(t.name, is_adult=(t.age >= 18))

        """
        if self.select_list is not None:
            raise excs.Error('Select list already specified')
        for name, _ in named_items.items():
            if not isinstance(name, str) or not is_valid_identifier(name):
                raise excs.Error(f'Invalid name: {name}')
        base_list = [(expr, None) for expr in items] + [(expr, k) for (k, expr) in named_items.items()]
        if len(base_list) == 0:
            return self

        # analyze select list; wrap literals with the corresponding expressions
        select_list: list[tuple[exprs.Expr, str | None]] = []
        for raw_expr, name in base_list:
            expr = exprs.Expr.from_object(raw_expr)
            if expr is None:
                raise excs.Error(f'Invalid expression: {raw_expr}')
            if expr.col_type.is_invalid_type() and not (isinstance(expr, exprs.Literal) and expr.val is None):
                raise excs.Error(f'Invalid type: {raw_expr}')
            if len(self._from_clause.tbls) == 1:
                # Select expressions need to be retargeted in order to handle snapshots correctly, as in expressions
                # such as `snapshot.select(base_tbl.col)`
                # TODO: For joins involving snapshots, we need a more sophisticated retarget() that can handle
                #     multiple TableVersionPaths.
                expr = expr.copy()
                try:
                    expr.retarget(self._from_clause.tbls[0])
                except Exception:
                    # If retarget() fails, then the succeeding is_bound_by() will raise an error.
                    pass
            if not expr.is_bound_by(self._from_clause.tbls):
                raise excs.Error(
                    f"That expression cannot be evaluated in the context of this query's tables "
                    f'({",".join(tbl.tbl_version.get().versioned_name for tbl in self._from_clause.tbls)}): {expr}'
                )
            select_list.append((expr, name))

        # check user provided names do not conflict among themselves or with auto-generated ones
        seen: set[str] = set()
        _, names = Query._normalize_select_list(self._from_clause.tbls, select_list)
        for name in names:
            if name in seen:
                repeated_names = [j for j, x in enumerate(names) if x == name]
                pretty = ', '.join(map(str, repeated_names))
                raise excs.Error(f'Repeated column name {name!r} in select() at positions: {pretty}')
            seen.add(name)

        return Query(
            from_clause=self._from_clause,
            select_list=select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
        )

    def where(self, pred: exprs.Expr) -> Query:
        """Filter rows based on a predicate.

        Args:
            pred: the predicate to filter rows

        Returns:
            A new Query with the specified predicates replacing the where-clause.

        Raises:
            Error: If the predicate is not a Pixeltable expression,
                or if it does not return a boolean value,
                or refers to tables not in the Query.

        Examples:
            Given the Query person from a table t with all its columns and rows:

            >>> person = t.select()

            Filter the above Query person to only include rows where the column 'age'
            (referenced in table t) is greater than 30:

            >>> query = person.where(t.age > 30)
        """
        if self.where_clause is not None:
            raise excs.Error('where() clause already specified')
        if self.sample_clause is not None:
            raise excs.Error('where() cannot be used after sample()')
        if not isinstance(pred, exprs.Expr):
            raise excs.Error(f'where() expects a Pixeltable expression; got: {pred}')
        if not pred.col_type.is_bool_type():
            raise excs.Error(f'where() expression needs to return `Bool`, but instead returns `{pred.col_type}`')
        return Query(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=pred,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
        )

    def _create_join_predicate(
        self, other: catalog.TableVersionPath, on: exprs.Expr | Sequence[exprs.ColumnRef]
    ) -> exprs.Expr:
        """Verifies user-specified 'on' argument and converts it into a join predicate."""
        col_refs: list[exprs.ColumnRef] = []
        joined_tbls = [*self._from_clause.tbls, other]

        if isinstance(on, exprs.ColumnRef):
            on = [on]
        elif isinstance(on, exprs.Expr):
            if not on.is_bound_by(joined_tbls):
                raise excs.Error(f'`on` expression cannot be evaluated in the context of the joined tables: {on}')
            if not on.col_type.is_bool_type():
                raise excs.Error(
                    f'`on` expects an expression of type `Bool`, but got one of type `{on.col_type}`: {on}'
                )
            return on
        elif not isinstance(on, Sequence) or len(on) == 0:
            raise excs.Error('`on` must be a sequence of column references or a boolean expression')

        assert isinstance(on, Sequence)
        for col_ref in on:
            if not isinstance(col_ref, exprs.ColumnRef):
                raise excs.Error('`on` must be a sequence of column references or a boolean expression')
            if not col_ref.is_bound_by(joined_tbls):
                raise excs.Error(f'`on` expression cannot be evaluated in the context of the joined tables: {col_ref}')
            col_refs.append(col_ref)

        predicates: list[exprs.Expr] = []
        # try to turn ColumnRefs into equality predicates
        assert len(col_refs) > 0 and len(joined_tbls) >= 2
        for col_ref in col_refs:
            # identify the referenced column by name in 'other'
            rhs_col = other.get_column(col_ref.col.name)
            if rhs_col is None:
                raise excs.Error(f'`on` column {col_ref.col.name!r} not found in joined table')
            rhs_col_ref = exprs.ColumnRef(rhs_col)

            lhs_col_ref: exprs.ColumnRef | None = None
            if any(tbl.has_column(col_ref.col) for tbl in self._from_clause.tbls):
                # col_ref comes from the existing from_clause, we use that directly
                lhs_col_ref = col_ref
            else:
                # col_ref comes from other, we need to look for a match in the existing from_clause by name
                for tbl in self._from_clause.tbls:
                    col = tbl.get_column(col_ref.col.name)
                    if col is None:
                        continue
                    if lhs_col_ref is not None:
                        raise excs.Error(f'`on`: ambiguous column reference: {col_ref.col.name}')
                    lhs_col_ref = exprs.ColumnRef(col)
                if lhs_col_ref is None:
                    tbl_names = [tbl.tbl_name() for tbl in self._from_clause.tbls]
                    raise excs.Error(f'`on`: column {col_ref.col.name!r} not found in any of: {" ".join(tbl_names)}')
            pred = exprs.Comparison(exprs.ComparisonOperator.EQ, lhs_col_ref, rhs_col_ref)
            predicates.append(pred)

        assert len(predicates) > 0
        if len(predicates) == 1:
            return predicates[0]
        else:
            return exprs.CompoundPredicate(operator=exprs.LogicalOperator.AND, operands=predicates)

    def join(
        self,
        other: catalog.Table,
        on: exprs.Expr | Sequence[exprs.ColumnRef] | None = None,
        how: plan.JoinType.LiteralType = 'inner',
    ) -> Query:
        """
        Join this Query with a table.

        Args:
            other: the table to join with
            on: the join condition, which can be either a) references to one or more columns or b) a boolean
                expression.

                - column references: implies an equality predicate that matches columns in both this
                    Query and `other` by name.

                    - column in `other`: A column with that same name must be present in this Query, and **it must
                        be unique** (otherwise the join is ambiguous).
                    - column in this Query: A column with that same name must be present in `other`.

                - boolean expression: The expressions must be valid in the context of the joined tables.
            how: the type of join to perform.

                - `'inner'`: only keep rows that have a match in both
                - `'left'`: keep all rows from this Query and only matching rows from the other table
                - `'right'`: keep all rows from the other table and only matching rows from this Query
                - `'full_outer'`: keep all rows from both this Query and the other table
                - `'cross'`: Cartesian product; no `on` condition allowed

        Returns:
            A new Query.

        Examples:
            Perform an inner join between t1 and t2 on the column id:

            >>> join1 = t1.join(t2, on=t2.id)

            Perform a left outer join of join1 with t3, also on id (note that we can't specify `on=t3.id` here,
            because that would be ambiguous, since both t1 and t2 have a column named id):

            >>> join2 = join1.join(t3, on=t2.id, how='left')

            Do the same, but now with an explicit join predicate:

            >>> join2 = join1.join(t3, on=t2.id == t3.id, how='left')

            Join t with d, which has a composite primary key (columns pk1 and pk2, with corresponding foreign
            key columns d1 and d2 in t):

            >>> query = t.join(d, on=(t.d1 == d.pk1) & (t.d2 == d.pk2), how='left')
        """
        if self.sample_clause is not None:
            raise excs.Error('join() cannot be used with sample()')
        join_pred: exprs.Expr | None
        if how == 'cross':
            if on is not None:
                raise excs.Error('`on` not allowed for cross join')
            join_pred = None
        else:
            if on is None:
                raise excs.Error(f'`how={how!r}` requires `on` to be present')
            join_pred = self._create_join_predicate(other._tbl_version_path, on)
        join_clause = plan.JoinClause(join_type=plan.JoinType.validated(how, '`how`'), join_predicate=join_pred)
        from_clause = plan.FromClause(
            tbls=[*self._from_clause.tbls, other._tbl_version_path],
            join_clauses=[*self._from_clause.join_clauses, join_clause],
        )
        return Query(
            from_clause=from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
        )

    def group_by(self, *grouping_items: Any) -> Query:
        """Add a group-by clause to this Query.

        Variants:
        - group_by(base_tbl): group a component view by their respective base table rows
        - group_by(expr1, expr2, expr3): group by the given expressions

        Note that grouping will be applied to the rows and take effect when
        used with an aggregation function like sum(), count() etc.

        Args:
            grouping_items: expressions to group by

        Returns:
            A new Query with the specified group-by clause.

        Raises:
            Error: If the group-by clause is already specified,
                or if the specified expression is invalid,
                or refer to tables not in the Query,
                or if the Query is a result of a join.

        Examples:
            Given the Query book from a table t with all its columns and rows:

            >>> book = t.select()

            Group the above Query book by the 'genre' column (referenced in table t):

            >>> query = book.group_by(t.genre)

            Use the above Query grouped by genre to count the number of
            books for each 'genre':

            >>> query = book.group_by(t.genre).select(t.genre, count=count(t.genre)).show()

            Use the above Query grouped by genre to the total price of
            books for each 'genre':

            >>> query = book.group_by(t.genre).select(t.genre, total=sum(t.price)).show()
        """
        if self.group_by_clause is not None:
            raise excs.Error('group_by() already specified')
        if self.sample_clause is not None:
            raise excs.Error('group_by() cannot be used with sample()')

        grouping_tbl: catalog.TableVersion | None = None
        group_by_clause: list[exprs.Expr] | None = None
        for item in grouping_items:
            if isinstance(item, (catalog.Table, catalog.TableVersion)):
                if len(grouping_items) > 1:
                    raise excs.Error('group_by(): only one Table can be specified')
                if len(self._from_clause.tbls) > 1:
                    raise excs.Error('group_by() with Table not supported for joins')
                grouping_tbl = item if isinstance(item, catalog.TableVersion) else item._tbl_version.get()
                # we need to make sure that the grouping table is a base of self.tbl
                base = self._first_tbl.find_tbl_version(grouping_tbl.id)
                if base is None or base.id == self._first_tbl.tbl_id:
                    raise excs.Error(
                        f'group_by(): {grouping_tbl.name!r} is not a base table of {self._first_tbl.tbl_name()!r}'
                    )
                break
            if not isinstance(item, exprs.Expr):
                raise excs.Error(f'Invalid expression in group_by(): {item}')
        if grouping_tbl is None:
            group_by_clause = list(grouping_items)
        return Query(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=group_by_clause,
            grouping_tbl=grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
        )

    def distinct(self) -> Query:
        """
        Remove duplicate rows from this Query.

        Note that grouping will be applied to the rows based on the select clause of this Query.
        In the absence of a select clause, by default, all columns are selected in the grouping.

        Examples:
            Select unique addresses from table `addresses`.

            >>> results = addresses.distinct()

            Select unique cities in table `addresses`

            >>> results = addresses.city.distinct()

            Select unique locations (street, city) in the state of `CA`

            >>> results = addresses.select(addresses.street, addresses.city).where(addresses.state == 'CA').distinct()
        """
        exps, _ = self._normalize_select_list(self._from_clause.tbls, self.select_list)
        return self.group_by(*exps)

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> Query:
        """Add an order-by clause to this Query.

        Args:
            expr_list: expressions to order by
            asc: whether to order in ascending order (True) or descending order (False).
                Default is True.

        Returns:
            A new Query with the specified order-by clause.

        Raises:
            Error: If the order-by clause is already specified,
                or if the specified expression is invalid,
                or refer to tables not in the Query.

        Examples:
            Given the Query book from a table t with all its columns and rows:

            >>> book = t.select()

            Order the above Query book by two columns (price, pages) in descending order:

            >>> query = book.order_by(t.price, t.pages, asc=False)

            Order the above Query book by price in descending order, but order the pages
            in ascending order:

            >>> query = book.order_by(t.price, asc=False).order_by(t.pages)
        """
        if self.sample_clause is not None:
            raise excs.Error('order_by() cannot be used with sample()')
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise excs.Error(f'Invalid expression in order_by(): {e}')
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend([(e.copy(), asc) for e in expr_list])
        return Query(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=order_by_clause,
            limit=self.limit_val,
        )

    def limit(self, n: int) -> Query:
        """Limit the number of rows in the Query.

        Args:
            n: Number of rows to select.

        Returns:
            A new Query with the specified limited rows.
        """
        if self.sample_clause is not None:
            raise excs.Error('limit() cannot be used with sample()')

        limit_expr = self._convert_param_to_typed_expr(n, ts.IntType(nullable=False), True, 'limit()')
        return Query(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=limit_expr,
        )

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> Query:
        """
        Return a new Query specifying a sample of rows from the Query, considered in a shuffled order.

        The size of the sample can be specified in three ways:

        - `n`: the total number of rows to produce as a sample
        - `n_per_stratum`: the number of rows to produce per stratum as a sample
        - `fraction`: the fraction of available rows to produce as a sample

        The sample can be stratified by one or more columns, which means that the sample will
        be selected from each stratum separately.

        The data is shuffled before creating the sample.

        Args:
            n: Total number of rows to produce as a sample.
            n_per_stratum: Number of rows to produce per stratum as a sample. This parameter is only valid if
                `stratify_by` is specified. Only one of `n` or `n_per_stratum` can be specified.
            fraction: Fraction of available rows to produce as a sample. This parameter is not usable with `n` or
                `n_per_stratum`. The fraction must be between 0.0 and 1.0.
            seed: Random seed for reproducible shuffling
            stratify_by: If specified, the sample will be stratified by these values.

        Returns:
            A new Query which specifies the sampled rows

        Examples:
            Given the Table `person` containing the field 'age', we can create samples of the table in various ways:

            Sample 100 rows from the above Table:

            >>> query = person.sample(n=100)

            Sample 10% of the rows from the above Table:

            >>> query = person.sample(fraction=0.1)

            Sample 10% of the rows from the above Table, stratified by the column 'age':

            >>> query = person.sample(fraction=0.1, stratify_by=t.age)

            Equal allocation sampling: Sample 2 rows from each age present in the above Table:

            >>> query = person.sample(n_per_stratum=2, stratify_by=t.age)

            Sampling is compatible with the where clause, so we can also sample from a filtered Query:

            >>> query = person.where(t.age > 30).sample(n=100)
        """
        # Check context of usage
        if self.sample_clause is not None:
            raise excs.Error('Multiple sample() clauses not allowed')
        if self.group_by_clause is not None:
            raise excs.Error('sample() cannot be used with group_by()')
        if self.order_by_clause is not None:
            raise excs.Error('sample() cannot be used with order_by()')
        if self.limit_val is not None:
            raise excs.Error('sample() cannot be used with limit()')
        if self._has_joins():
            raise excs.Error('sample() cannot be used with join()')

        # Check paramter combinations
        if (n is not None) + (n_per_stratum is not None) + (fraction is not None) != 1:
            raise excs.Error('Exactly one of `n`, `n_per_stratum`, or `fraction` must be specified.')
        if n_per_stratum is not None and stratify_by is None:
            raise excs.Error('Must specify `stratify_by` to use `n_per_stratum`')

        # Check parameter types and values
        n = self.validate_constant_type_range(n, ts.IntType(nullable=False), False, 'n', (1, None))
        n_per_stratum = self.validate_constant_type_range(
            n_per_stratum, ts.IntType(nullable=False), False, 'n_per_stratum', (1, None)
        )
        fraction = self.validate_constant_type_range(
            fraction, ts.FloatType(nullable=False), False, 'fraction', (0.0, 1.0)
        )
        seed = self.validate_constant_type_range(seed, ts.IntType(nullable=False), False, 'seed')

        # analyze stratify list
        stratify_exprs: list[exprs.Expr] = []
        if stratify_by is not None:
            if isinstance(stratify_by, exprs.Expr):
                stratify_by = [stratify_by]
            if not isinstance(stratify_by, (list, tuple)):
                raise excs.Error('`stratify_by` must be a list of scalar expressions')
            for expr in stratify_by:
                if expr is None or not isinstance(expr, exprs.Expr):
                    raise excs.Error(f'Invalid expression: {expr}')
                if not expr.col_type.is_scalar_type():
                    raise excs.Error(f'Invalid type: expression must be a scalar type (not `{expr.col_type}`)')
                if not expr.is_bound_by(self._from_clause.tbls):
                    raise excs.Error(
                        f"That expression cannot be evaluated in the context of this query's tables "
                        f'({",".join(tbl.tbl_name() for tbl in self._from_clause.tbls)}): {expr}'
                    )
                stratify_exprs.append(expr)

        sample_clause = SampleClause(None, n, n_per_stratum, fraction, seed, stratify_exprs)

        return Query(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl=self.grouping_tbl,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            sample_clause=sample_clause,
        )

    def update(self, value_spec: dict[str, Any], cascade: bool = True) -> UpdateStatus:
        """Update rows in the underlying table of the Query.

        Update rows in the table with the specified value_spec.

        Args:
            value_spec: a dict of column names to update and the new value to update it to.
            cascade: if True, also update all computed columns that transitively depend
                    on the updated columns, including within views. Default is True.

        Returns:
            UpdateStatus: the status of the update operation.

        Example:
            Given the Query person from a table t with all its columns and rows:

            >>> person = t.select()

            Via the above Query person, update the column 'city' to 'Oakland'
            and 'state' to 'CA' in the table t:

            >>> person.update({'city': 'Oakland', 'state': 'CA'})

            Via the above Query person, update the column 'age' to 30 for any
            rows where 'year' is 2014 in the table t:

            >>> person.where(t.year == 2014).update({'age': 30})
        """
        self._validate_mutable('update', False)
        with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=True, lock_mutable_tree=True):
            return self._first_tbl.tbl_version.get().update(value_spec, where=self.where_clause, cascade=cascade)

    def recompute_columns(
        self, *columns: str | exprs.ColumnRef, errors_only: bool = False, cascade: bool = True
    ) -> UpdateStatus:
        """Recompute one or more computed columns of the underlying table of the Query.

        Args:
            columns: The names or references of the computed columns to recompute.
            errors_only: If True, only run the recomputation for rows that have errors in the column (ie, the column's
                `errortype` property indicates that an error occurred). Only allowed for recomputing a single column.
            cascade: if True, also update all computed columns that transitively depend on the recomputed columns.

        Returns:
            UpdateStatus: the status of the operation.

        Example:
            For table `person` with column `age` and computed column `height`, recompute the value of `height` for all
            rows where `age` is less than 18:

            >>> query = person.where(t.age < 18).recompute_columns(person.height)
        """
        self._validate_mutable('recompute_columns', False)
        with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=True, lock_mutable_tree=True):
            tbl = Catalog.get().get_table_by_id(self._first_tbl.tbl_id)
            return tbl.recompute_columns(*columns, where=self.where_clause, errors_only=errors_only, cascade=cascade)

    def delete(self) -> UpdateStatus:
        """Delete rows form the underlying table of the Query.

        The delete operation is only allowed for Queries on base tables.

        Returns:
            UpdateStatus: the status of the delete operation.

        Example:
            For a table `person` with column `age`, delete all rows where 'age' is less than 18:

            >>> person.where(t.age < 18).delete()
        """
        self._validate_mutable('delete', False)
        if not self._first_tbl.is_insertable():
            raise excs.Error('Cannot use `delete` on a view.')
        with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=True, lock_mutable_tree=True):
            return self._first_tbl.tbl_version.get().delete(where=self.where_clause)

    def _validate_mutable(self, op_name: str, allow_select: bool) -> None:
        """Tests whether this Query can be mutated (such as by an update operation).

        Args:
            op_name: The name of the operation for which the test is being performed.
            allow_select: If True, allow a select() specification in the Query.
        """
        self._validate_mutable_op_sequence(op_name, allow_select)

        # TODO: Reconcile these with Table.__check_mutable()
        assert len(self._from_clause.tbls) == 1
        # First check if it's a replica, since every replica handle is also a snapshot
        if self._first_tbl.is_replica():
            raise excs.Error(f'Cannot use `{op_name}` on a replica.')
        if self._first_tbl.is_snapshot():
            raise excs.Error(f'Cannot use `{op_name}` on a snapshot.')

    def _validate_mutable_op_sequence(self, op_name: str, allow_select: bool) -> None:
        """Tests whether the sequence of operations on this Query is valid for a mutation operation."""
        if self.group_by_clause is not None or self.grouping_tbl is not None:
            raise excs.Error(f'Cannot use `{op_name}` after `group_by`.')
        if self.order_by_clause is not None:
            raise excs.Error(f'Cannot use `{op_name}` after `order_by`.')
        if self.select_list is not None and not allow_select:
            raise excs.Error(f'Cannot use `{op_name}` after `select`.')
        if self.limit_val is not None:
            raise excs.Error(f'Cannot use `{op_name}` after `limit`.')
        if self._has_joins():
            raise excs.Error(f'Cannot use `{op_name}` after `join`.')

    def as_dict(self) -> dict[str, Any]:
        """
        Returns:
            Dictionary representing this Query.
        """
        d = {
            '_classname': 'Query',
            'from_clause': {
                'tbls': [tbl.as_dict() for tbl in self._from_clause.tbls],
                'join_clauses': [dataclasses.asdict(clause) for clause in self._from_clause.join_clauses],
            },
            'select_list': [(e.as_dict(), name) for (e, name) in self.select_list]
            if self.select_list is not None
            else None,
            'where_clause': self.where_clause.as_dict() if self.where_clause is not None else None,
            'group_by_clause': [e.as_dict() for e in self.group_by_clause]
            if self.group_by_clause is not None
            else None,
            'grouping_tbl': self.grouping_tbl.as_dict() if self.grouping_tbl is not None else None,
            'order_by_clause': [(e.as_dict(), asc) for (e, asc) in self.order_by_clause]
            if self.order_by_clause is not None
            else None,
            'limit_val': self.limit_val.as_dict() if self.limit_val is not None else None,
            'sample_clause': self.sample_clause.as_dict() if self.sample_clause is not None else None,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'Query':
        # we need to wrap the construction with a transaction, because it might need to load metadata
        with Catalog.get().begin_xact(for_write=False):
            tbls = [catalog.TableVersionPath.from_dict(tbl_dict) for tbl_dict in d['from_clause']['tbls']]
            join_clauses = [plan.JoinClause(**clause_dict) for clause_dict in d['from_clause']['join_clauses']]
            from_clause = plan.FromClause(tbls=tbls, join_clauses=join_clauses)
            select_list = (
                [(exprs.Expr.from_dict(e), name) for e, name in d['select_list']]
                if d['select_list'] is not None
                else None
            )
            where_clause = exprs.Expr.from_dict(d['where_clause']) if d['where_clause'] is not None else None
            group_by_clause = (
                [exprs.Expr.from_dict(e) for e in d['group_by_clause']] if d['group_by_clause'] is not None else None
            )
            grouping_tbl = catalog.TableVersion.from_dict(d['grouping_tbl']) if d['grouping_tbl'] is not None else None
            order_by_clause = (
                [(exprs.Expr.from_dict(e), asc) for e, asc in d['order_by_clause']]
                if d['order_by_clause'] is not None
                else None
            )
            limit_val = exprs.Expr.from_dict(d['limit_val']) if d['limit_val'] is not None else None
            sample_clause = SampleClause.from_dict(d['sample_clause']) if d['sample_clause'] is not None else None

            return Query(
                from_clause=from_clause,
                select_list=select_list,
                where_clause=where_clause,
                group_by_clause=group_by_clause,
                grouping_tbl=grouping_tbl,
                order_by_clause=order_by_clause,
                limit=limit_val,
                sample_clause=sample_clause,
            )

    def _hash_result_set(self) -> str:
        """Return a hash that changes when the result set changes."""
        d = self.as_dict()
        # add list of referenced table versions (the actual versions, not the effective ones) in order to force cache
        # invalidation when any of the referenced tables changes
        d['tbl_versions'] = [
            tbl_version.get().version for tbl in self._from_clause.tbls for tbl_version in tbl.get_tbl_versions()
        ]
        summary_string = json.dumps(d)
        return hashlib.sha256(summary_string.encode()).hexdigest()

    def to_coco_dataset(self) -> Path:
        """Convert the Query to a COCO dataset.
        This Query must return a single json-typed output column in the following format:

        ```python
        {
            'image': PIL.Image.Image,
            'annotations': [
                {
                    'bbox': [x: int, y: int, w: int, h: int],
                    'category': str | int,
                },
                ...
            ],
        }
        ```

        Returns:
            Path to the COCO dataset file.
        """
        from pixeltable.utils.coco import write_coco_dataset

        cache_key = self._hash_result_set()
        dest_path = Env.get().dataset_cache_dir / f'coco_{cache_key}'
        if dest_path.exists():
            assert dest_path.is_dir()
            data_file_path = dest_path / 'data.json'
            assert data_file_path.exists()
            assert data_file_path.is_file()
            return data_file_path
        else:
            # TODO: extend begin_xact() to accept multiple TVPs for joins
            with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=False):
                return write_coco_dataset(self, dest_path)

    def to_pytorch_dataset(self, image_format: str = 'pt') -> 'torch.utils.data.IterableDataset':
        """
        Convert the Query to a pytorch IterableDataset suitable for parallel loading
        with torch.utils.data.DataLoader.

        This method requires pyarrow >= 13, torch and torchvision to work.

        This method serializes data so it can be read from disk efficiently and repeatedly without
        re-executing the query. This data is cached to disk for future re-use.

        Args:
            image_format: format of the images. Can be 'pt' (pytorch tensor) or 'np' (numpy array).
                    'np' means image columns return as an RGB uint8 array of shape HxWxC.
                    'pt' means image columns return as a CxHxW tensor with values in [0,1] and type torch.float32.
                        (the format output by torchvision.transforms.ToTensor())

        Returns:
            A pytorch IterableDataset: Columns become fields of the dataset, where rows are returned as a dictionary
                compatible with torch.utils.data.DataLoader default collation.

        Constraints:
            The default collate_fn for torch.data.util.DataLoader cannot represent null values as part of a
            pytorch tensor when forming batches. These values will raise an exception while running the dataloader.

            If you have them, you can work around None values by providing your custom collate_fn to the DataLoader
            (and have your model handle it). Or, if these are not meaningful values within a minibtach, you can
            modify or remove any such values through selections and filters prior to calling to_pytorch_dataset().
        """
        # check dependencies
        Env.get().require_package('pyarrow', [13])
        Env.get().require_package('torch')
        Env.get().require_package('torchvision')

        from pixeltable.io import export_parquet
        from pixeltable.utils.pytorch import PixeltablePytorchDataset

        cache_key = self._hash_result_set()

        dest_path = (Env.get().dataset_cache_dir / f'df_{cache_key}').with_suffix('.parquet')
        if dest_path.exists():  # fast path: use cache
            assert dest_path.is_dir()
        else:
            with Catalog.get().begin_xact(tbl=self._first_tbl, for_write=False):
                export_parquet(self, dest_path, inline_images=True)

        return PixeltablePytorchDataset(path=dest_path, image_format=image_format)

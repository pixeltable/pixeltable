"""The shape of a query: the clauses it carries and the operations that build them.

Separate from _query.py so that a subclass defined in another package (catalog.model.query.ModelQuery) can
import its base without importing Query, which would close an import cycle.
"""

from __future__ import annotations

import builtins
import copy
import itertools
from abc import ABC
from typing import Any, Self, Sequence, cast, overload
from uuid import UUID

import pandas as pd

from pixeltable import catalog, exceptions as excs, exprs, type_system as ts
from pixeltable.catalog import is_valid_identifier
from pixeltable.query_clauses import FromClause, JoinClause, JoinType, SampleClause
from pixeltable.runtime import get_runtime
from pixeltable.type_system import ColumnType
from pixeltable.utils.description_helper import DescriptionHelper

__all__ = ['QueryBase']


class QueryBase(ABC):
    """The shape of a query: its from-clause and the clauses layered on it, and the operations that build them.

    A subclass adds what it means to run the query. Query does that against tables in a catalog; ModelQuery
    stands for a query over a model that is not bound to a table yet, and runs nothing.

    Thread-safe.
    """

    # immutable after init()
    _from_clause: FromClause

    select_list: list[tuple[exprs.Expr, str | None]] | None
    # construction-time snapshot of the resolved select list; None if select_list is None
    _select_list_exprs: list[exprs.Expr] | None
    # construction-time snapshot of the resolved schema; None if select_list is None
    _schema: dict[str, ts.ColumnType] | None

    where_clause: exprs.Expr | None
    group_by_clause: list[exprs.Expr] | None
    grouping_tbl_key: catalog.TableVersionKey | None
    order_by_clause: list[tuple[exprs.Expr, bool]] | None
    limit_val: exprs.Expr | None
    offset_val: exprs.Expr | None
    sample_clause: SampleClause | None

    # IDs of all tables referenced by this query (from-clause path + exprs). Computed once on
    # first access, then cached: the value depends on the static query shape and never changes.
    _referenced_tbl_ids: set[UUID] | None

    def __init__(
        self,
        from_clause: FromClause | None = None,
        select_list: list[tuple[exprs.Expr, str | None]] | None = None,
        where_clause: exprs.Expr | None = None,
        group_by_clause: list[exprs.Expr] | None = None,
        grouping_tbl_key: catalog.TableVersionKey | None = None,
        order_by_clause: list[tuple[exprs.Expr, bool]] | None = None,  # list[(expr, asc)]
        limit: exprs.Expr | None = None,
        offset: exprs.Expr | None = None,
        sample_clause: SampleClause | None = None,
    ):
        self._from_clause = from_clause

        if from_clause is not None and from_clause.is_local:
            # find out about dropped tables early; only applicable for local tables
            get_runtime().catalog.validate_tbls_exist({t.tbl_id for t in from_clause.tbls})

        # exprs contain execution state and therefore cannot be shared
        self.select_list = copy.deepcopy(select_list)
        if self.select_list is None:
            self._select_list_exprs = None
            self._schema = None
        else:
            select_list_exprs, column_names = self._normalize_select_list(self._from_clause.tbls, self.select_list)
            # check select list after expansion to catch early
            assert len(column_names) == len(select_list_exprs)
            self._select_list_exprs = select_list_exprs
            self._schema = {column_names[i]: select_list_exprs[i].col_type for i in range(len(column_names))}

        self.where_clause = copy.deepcopy(where_clause)
        assert group_by_clause is None or grouping_tbl_key is None
        self.group_by_clause = copy.deepcopy(group_by_clause)
        self.grouping_tbl_key = grouping_tbl_key
        self.order_by_clause = copy.deepcopy(order_by_clause)
        self.limit_val = limit
        self.offset_val = offset
        self.sample_clause = sample_clause
        self._referenced_tbl_ids = None

    @classmethod
    def _normalize_select_list(
        cls, tbls: list[catalog.TablePath], select_list: list[tuple[exprs.Expr, str | None]] | None
    ) -> tuple[list[exprs.Expr], list[str]]:
        """
        Expand select list information with all columns and their names
        Returns:
            a pair composed of the list of expressions and the list of corresponding names
        """
        if select_list is None:
            select_list = [
                (exprs.ColumnRef(col_md, tbl.is_validate_on_read(col_md)), None)
                for tbl in tbls
                for col_md in tbl.column_md()
            ]

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
    def has_select_list(self) -> bool:
        """Returns True if the query has an explicit select list (constructed with `select()`)."""
        return self.select_list is not None

    @property
    def _first_tbl(self) -> catalog.TableVersionPath:
        assert self._from_clause.is_local
        return self._from_clause.tvps[0]

    @property
    def _effective_select_list(self) -> list[tuple[exprs.Expr, str]]:
        """Return the select list that would get materialized by collect()."""
        if self.select_list is not None:
            return list(zip(self._select_list_exprs, self._schema.keys()))
        # SELECT * case: re-resolve against the current table schema
        select_list_exprs, column_names = self._normalize_select_list(self._from_clause.tbls, None)
        return list(zip(select_list_exprs, column_names))

    def _vars(self) -> dict[str, exprs.Variable]:
        """
        Return a dict mapping variable name to Variable for all Variables contained in any component of the Query
        """
        all_exprs: list[exprs.Expr] = []
        if self._select_list_exprs is not None:
            # _select_list_exprs is None: no Variables without an explicit select list
            all_exprs.extend(self._select_list_exprs)
        if self.where_clause is not None:
            all_exprs.append(self.where_clause)
        if self.group_by_clause is not None:
            all_exprs.extend(self.group_by_clause)
        if self.order_by_clause is not None:
            all_exprs.extend([expr for expr, _ in self.order_by_clause])
        if self.limit_val is not None:
            all_exprs.append(self.limit_val)
        if self.offset_val is not None:
            all_exprs.append(self.offset_val)
        vars = exprs.Expr.list_subexprs(all_exprs, expr_class=exprs.Variable)
        unique_vars: dict[str, exprs.Variable] = {}
        for var in vars:
            if var.name not in unique_vars:
                unique_vars[var.name] = var
            elif unique_vars[var.name].col_type != var.col_type:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'Multiple definitions of parameter {var.name!r}'
                )
        return unique_vars

    def _component_exprs(self) -> list[exprs.Expr]:
        """Returns all exprs referenced in this query's clauses"""
        result: list[exprs.Expr] = []
        if self._select_list_exprs is not None:
            result.extend(self._select_list_exprs)
        if self.where_clause is not None:
            result.append(self.where_clause)
        if self.group_by_clause is not None:
            result.extend(self.group_by_clause)
        if self.order_by_clause is not None:
            result.extend(expr for expr, _ in self.order_by_clause)
        if self.limit_val is not None:
            result.append(self.limit_val)
        if self.offset_val is not None:
            result.append(self.offset_val)
        if self.sample_clause is not None and self.sample_clause.stratify_exprs is not None:
            result.extend(self.sample_clause.stratify_exprs)
        if self._from_clause is not None:
            result.extend(c.join_predicate for c in self._from_clause.join_clauses if c.join_predicate is not None)
        return result

    @classmethod
    def _convert_param_to_typed_expr(
        cls, v: Any, required_type: ts.ColumnType, required: bool, name: str, range: tuple[Any, Any] | None = None
    ) -> exprs.Expr | None:
        if v is None:
            if required:
                raise excs.RequestError(excs.ErrorCode.MISSING_REQUIRED, f'{name!r} parameter must be present')
            return v
        v_expr = exprs.Expr.from_object(v)
        if not v_expr.col_type.matches(required_type):
            raise excs.RequestError(
                excs.ErrorCode.TYPE_MISMATCH,
                f'{name!r} parameter must be of type `{required_type}`; got `{v_expr.col_type}`',
            )
        if range is not None:
            if not isinstance(v_expr, exprs.Literal):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} parameter must be a constant; got: {v_expr}'
                )
            if range[0] is not None and not (v_expr.val >= range[0]):
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} parameter must be >= {range[0]}')
            if range[1] is not None and not (v_expr.val <= range[1]):
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{name!r} parameter must be <= {range[1]}')
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

    def _has_joins(self) -> bool:
        return len(self._from_clause.join_clauses) > 0

    @property
    def schema(self) -> dict[str, ColumnType]:
        """Column names and types in this Query."""
        if self.select_list is not None:
            return self._schema
        else:
            # need to re-resolve select list
            return {name: e.col_type for e, name in self._effective_select_list}

    def _replace_select_list(self, new_exprs: list[exprs.Expr]) -> Self:
        """Return a new query with the given select-list exprs.

        All other clauses are cloned either here or in __init__().
        """
        select_list = self._effective_select_list
        assert len(new_exprs) == len(select_list)
        select_list = [(e, n) for e, (_, n) in zip(new_exprs, select_list)]
        return type(self)(
            from_clause=self._from_clause,
            select_list=select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=copy.deepcopy(self.limit_val),
            offset=copy.deepcopy(self.offset_val),
            sample_clause=copy.deepcopy(self.sample_clause),
        )

    def referenced_tbl_ids(self) -> set[UUID]:
        """Returns the IDs of all tables referenced by this query.

        Walks the query's static structure (exprs + from-clause path) on first call and caches
        the result; the value depends on the static query shape and never changes.
        """
        if self._referenced_tbl_ids is not None:
            return self._referenced_tbl_ids

        all_exprs = itertools.chain(
            # _select_list_exprs is None: no external ColumnRefs without an explicit select list
            self._select_list_exprs or [],
            [] if self.where_clause is None else [self.where_clause],
            self.group_by_clause or [],
            [] if self.order_by_clause is None else (e for e, _ in self.order_by_clause),
        )
        tbl_ids = exprs.Expr.list_tbl_ids(all_exprs)
        for tp in self._from_clause.tbls:
            tbl_ids.update(tp.tbl_ids)

        self._referenced_tbl_ids = tbl_ids
        return tbl_ids

    def _descriptors(self) -> DescriptionHelper:
        helper = DescriptionHelper()
        helper.append(self._col_descriptor())
        qd = self._query_descriptor()
        if not qd.empty:
            helper.append(qd, show_index=True, show_header=False)
        return helper

    def _col_descriptor(self) -> pd.DataFrame:
        select_list = self._effective_select_list
        return pd.DataFrame(
            [
                {'Name': name, 'Type': repr(expr.col_type), 'Expression': expr.display_str(inline=False)}
                for expr, name in select_list
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
            limit_str = self.limit_val.display_str(inline=False)
            if self.offset_val is not None:
                limit_str += f',{self.offset_val.display_str(inline=False)}'
            info_vals.append(limit_str)
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

    def select(self, *items: Any, **named_items: Any) -> Self:
        """Select columns or expressions from the Query.

        Args:
            items: expressions to be selected
            named_items: named expressions to be selected

        Returns:
            A new query with the specified select list.

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
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'Select list already specified')
        for name, _ in named_items.items():
            if not isinstance(name, str) or not is_valid_identifier(name):
                raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'Invalid name: {name}')
        base_list = [(expr, None) for expr in items] + [(expr, k) for (k, expr) in named_items.items()]
        if len(base_list) == 0:
            return self

        # analyze select list; wrap literals with the corresponding expressions
        select_list: list[tuple[exprs.Expr, str | None]] = []
        for raw_expr, name in base_list:
            expr = exprs.Expr.from_object(raw_expr)
            if expr is None:
                raise excs.RequestError(excs.ErrorCode.INVALID_EXPRESSION, f'Invalid expression: {raw_expr}')
            if expr.col_type.is_invalid_type() and not (isinstance(expr, exprs.Literal) and expr.val is None):
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, f'Invalid type: {raw_expr}')
            if len(self._from_clause.tbls) == 1 and self._from_clause.is_local:
                # Select expressions need to be retargeted in order to handle snapshots correctly, as in expressions
                # such as `snapshot.select(base_tbl.col)`
                # TODO: For joins involving snapshots, we need a more sophisticated retarget() that can handle
                #     multiple TableVersionPaths.
                expr = expr.copy()
                try:
                    expr = expr.retarget_path(self._from_clause.tvps[0])
                except Exception:
                    # If retarget() fails, then the succeeding is_bound_by() will raise an error.
                    pass
            if not expr.is_bound_by(self._from_clause.tbls):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f"That expression cannot be evaluated in the context of this query's tables "
                    f'({", ".join(tbl.tbl_name() for tbl in self._from_clause.tbls)}): {expr}',
                )
            select_list.append((expr, name))

        # check user provided names do not conflict among themselves or with auto-generated ones
        seen: set[str] = set()
        _, names = self._normalize_select_list(self._from_clause.tbls, select_list)
        for name in names:
            if name in seen:
                repeated_names = [j for j, x in enumerate(names) if x == name]
                pretty = ', '.join(map(str, repeated_names))
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'Repeated column name {name!r} in select() at positions: {pretty}',
                )
            seen.add(name)

        return type(self)(
            from_clause=self._from_clause,
            select_list=select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
        )

    def where(self, pred: exprs.Expr) -> Self:
        """Filter rows based on a predicate.

        Args:
            pred: the predicate to filter rows

        Returns:
            A new query with the specified predicates replacing the where-clause.

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
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'where() clause already specified')
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'where() cannot be used after sample()')
        if not isinstance(pred, exprs.Expr):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_EXPRESSION, f'where() expects a Pixeltable expression; got: {pred}'
            )
        if not pred.col_type.is_bool_type():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                f'where() expression needs to return `Bool`, but instead returns `{pred.col_type}`',
            )
        return type(self)(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=pred,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
        )

    def _create_join_predicate(
        self, other: catalog.TablePath, on: exprs.Expr | Sequence[exprs.ColumnRef]
    ) -> exprs.Expr:
        """Verifies user-specified 'on' argument and converts it into a join predicate."""
        col_refs: list[exprs.ColumnRef] = []
        joined_tbls = [*self._from_clause.tbls, other]

        if isinstance(on, exprs.ColumnRef):
            on = [on]
        elif isinstance(on, exprs.Expr):
            if not on.is_bound_by(joined_tbls):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'`on` expression cannot be evaluated in the context of the joined tables: {on}',
                )
            if not on.col_type.is_bool_type():
                raise excs.RequestError(
                    excs.ErrorCode.TYPE_MISMATCH,
                    f'`on` expects an expression of type `Bool`, but got one of type `{on.col_type}`: {on}',
                )
            return on
        elif not isinstance(on, Sequence) or len(on) == 0:
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT, '`on` must be a sequence of column references or a boolean expression'
            )

        assert isinstance(on, Sequence)
        for col_ref in on:
            if not isinstance(col_ref, exprs.ColumnRef):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    '`on` must be a sequence of column references or a boolean expression',
                )
            if not col_ref.is_bound_by(joined_tbls):
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'`on` expression cannot be evaluated in the context of the joined tables: {col_ref}',
                )
            col_refs.append(col_ref)

        predicates: list[exprs.Expr] = []
        # try to turn ColumnRefs into equality predicates
        assert len(col_refs) > 0 and len(joined_tbls) >= 2
        for col_ref in col_refs:
            # identify the referenced column by name in 'other'
            col_name = col_ref.col_md.name
            rhs_col_md = other.get_column_md_by_name(col_name)
            if rhs_col_md is None:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND, f'`on` column {col_name!r} not found in joined table'
                )
            rhs_col_ref = exprs.ColumnRef(rhs_col_md)

            lhs_col_ref: exprs.ColumnRef | None = None
            if any(tbl.has_column(col_ref.col_md.qcolid) for tbl in self._from_clause.tbls):
                # col_ref comes from the existing from_clause, we use that directly
                lhs_col_ref = col_ref
            else:
                # col_ref comes from other, we need to look for a match in the existing from_clause by name
                for tbl in self._from_clause.tbls:
                    col_md = tbl.get_column_md_by_name(col_name)
                    if col_md is None:
                        continue
                    if lhs_col_ref is not None:
                        raise excs.RequestError(
                            excs.ErrorCode.UNSUPPORTED_OPERATION, f'`on`: ambiguous column reference: {col_name!r}'
                        )
                    lhs_col_ref = exprs.ColumnRef(col_md)
                if lhs_col_ref is None:
                    tbl_names = [tbl.tbl_name() for tbl in self._from_clause.tbls]
                    raise excs.NotFoundError(
                        excs.ErrorCode.COLUMN_NOT_FOUND,
                        f'`on`: column {col_name!r} not found in any of: {" ".join(tbl_names)}',
                    )
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
        how: JoinType.LiteralType = 'inner',
    ) -> Self:
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
            A new query.

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
        assert len(self._from_clause.tbls) > 0
        # a join mixing catalogs (e.g. local + hosted) is rejected by FromClause's same-catalog check below
        if self._from_clause.tbls[0].is_data_versioned() != other._is_data_versioned():
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION,
                'join is not supported between data-versioned and operational tables',
            )
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'join() cannot be used with sample()')
        join_pred: exprs.Expr | None
        if how == 'cross':
            if on is not None:
                raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, '`on` not allowed for cross join')
            join_pred = None
        else:
            if on is None:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION, f'`how={how!r}` requires `on` to be present'
                )
            join_pred = self._create_join_predicate(other._tbl_path, on)
        join_clause = JoinClause(join_type=JoinType.validated(how, '`how`'), join_predicate=join_pred)
        from_clause = FromClause(
            tbls=[*self._from_clause.tbls, other._tbl_path], join_clauses=[*self._from_clause.join_clauses, join_clause]
        )
        return type(self)(
            from_clause=from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
        )

    @overload
    def group_by(self, grouping_tbl: catalog.Table, /) -> Self:
        """Group a component view by its base table's rows."""

    @overload
    def group_by(self, *grouping_items: exprs.Expr) -> Self:
        """Group by the given expressions."""

    def group_by(self, *grouping_items: exprs.Expr | catalog.Table) -> Self:
        """Add a group-by clause to this Query.

        Variants:
        - group_by(base_tbl): group a component view by their respective base table rows
        - group_by(expr1, expr2, expr3): group by the given expressions

        Note that grouping will be applied to the rows and take effect when
        used with an aggregation function like sum(), count() etc.

        Args:
            grouping_items: expressions to group by

        Returns:
            A new query with the specified group-by clause.

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

            >>> query = (
            ...     book.group_by(t.genre).select(t.genre, count=count(t.genre)).show()
            ... )

            Use the above Query grouped by genre to the total price of
            books for each 'genre':

            >>> query = book.group_by(t.genre).select(t.genre, total=sum(t.price)).show()
        """
        if self.group_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.INVALID_STATE, 'group_by() already specified')
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'group_by() cannot be used with sample()')

        grouping_tbl_key: catalog.TableVersionKey | None = None
        group_by_clause: list[exprs.Expr] | None = None
        for item in grouping_items:
            if isinstance(item, catalog.Table):
                if len(grouping_items) > 1:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, 'group_by(): only one Table can be specified'
                    )
                if len(self._from_clause.tbls) > 1:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION, 'group_by() with Table not supported for joins'
                    )
                # the grouping table must be a base of this query's table (and not the table itself)
                from_path = self._from_clause.tbls[0]
                grouping_path = item._tbl_path
                grouping_tbl_key = from_path.find_tbl_version(grouping_path.tbl_id)
                if grouping_tbl_key is None or grouping_tbl_key.tbl_id == from_path.tbl_id:
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f'group_by(): {grouping_path.tbl_name()!r} is not a base table of {from_path.tbl_name()!r}',
                    )
                break
            if not isinstance(item, exprs.Expr):
                raise excs.RequestError(excs.ErrorCode.INVALID_EXPRESSION, f'Invalid expression in group_by(): {item}')
        if grouping_tbl_key is None:
            # no Table item was found, so every item passed the Expr check above
            group_by_clause = cast('list[exprs.Expr]', list(grouping_items))
        return type(self)(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=group_by_clause,
            grouping_tbl_key=grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
        )

    def distinct(self) -> Self:
        """
        Remove duplicate rows from this Query.

        Note that grouping will be applied to the rows based on the select clause of this Query.
        In the absence of a select clause, by default, all columns are selected in the grouping.

        Examples:
            Select unique addresses from table `addresses`.

            >>> results = addresses.distinct()

            Select unique cities in table `addresses`

            >>> results = addresses.select(addresses.city).distinct()

            Select unique locations (street, city) in the state of `CA`

            >>> results = (
            ...     addresses.select(addresses.street, addresses.city)
            ...     .where(addresses.state == 'CA')
            ...     .distinct()
            ... )
        """
        exps, _ = self._normalize_select_list(self._from_clause.tbls, self.select_list)
        return self.group_by(*exps)

    def order_by(self, *expr_list: exprs.Expr, asc: bool = True) -> Self:
        """Add an order-by clause to this Query.

        Args:
            expr_list: expressions to order by
            asc: whether to order in ascending order (True) or descending order (False).
                Default is True.

        Returns:
            A new query with the specified order-by clause.

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
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'order_by() cannot be used with sample()')
        for e in expr_list:
            if not isinstance(e, exprs.Expr):
                raise excs.RequestError(excs.ErrorCode.INVALID_EXPRESSION, f'Invalid expression in order_by(): {e}')
        order_by_clause = self.order_by_clause if self.order_by_clause is not None else []
        order_by_clause.extend((e.copy(), asc) for e in expr_list)
        return type(self)(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
        )

    def limit(self, n: int, offset: int | None = None) -> Self:
        """Limit the number of rows in the Query, optionally skipping rows for pagination.

        Args:
            n: Number of rows to select.
            offset: Number of rows to skip before returning results. Default is None (no offset).

        Returns:
            A new query with the specified limited rows.

        Examples:
            >>> query = t.select()

            Get the first 10 rows:

            >>> query.limit(10).collect()

            Get rows 21-30 (skip first 20, return next 10):

            >>> query.limit(10, offset=20).collect()
        """
        if self.sample_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'limit() cannot be used with sample()')

        # Reject negative int constants here. Non-int types fall through to _convert_param_to_typed_expr,
        # which raises TYPE_MISMATCH. Expression-valued limits (from @pxt.query bodies) aren't validated
        # here; users constructing queries directly always pass a Python int.
        if isinstance(n, int) and not isinstance(n, bool) and n < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, "'limit()' parameter must be >= 0")
        if offset is not None and isinstance(offset, int) and not isinstance(offset, bool) and offset < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, "'offset' parameter must be >= 0")

        limit_expr = self._convert_param_to_typed_expr(n, ts.IntType(nullable=False), True, 'limit()')
        if not isinstance(limit_expr, (exprs.Literal, exprs.Variable)):
            raise excs.RequestError(
                excs.ErrorCode.INVALID_ARGUMENT,
                f'limit(): parameter must be an int constant or query parameter; got: {n}',
            )
        offset_expr = None
        if offset is not None:
            offset_expr = self._convert_param_to_typed_expr(offset, ts.IntType(nullable=False), False, 'offset')
            if not isinstance(offset_expr, (exprs.Literal, exprs.Variable)):
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    f'offset: parameter must be an int constant or query parameter; got: {offset}',
                )

        return type(self)(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=limit_expr,
            offset=offset_expr,
        )

    def sample(
        self,
        n: int | None = None,
        n_per_stratum: int | None = None,
        fraction: float | None = None,
        seed: int | None = None,
        stratify_by: Any = None,
    ) -> Self:
        """
        Return a new query specifying a sample of rows from this query, considered in a shuffled order.

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
            A new query which specifies the sampled rows

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
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'Multiple sample() clauses not allowed')
        if self.group_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'sample() cannot be used with group_by()')
        if self.order_by_clause is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'sample() cannot be used with order_by()')
        if self.limit_val is not None:
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'sample() cannot be used with limit()')
        if self._has_joins():
            raise excs.RequestError(excs.ErrorCode.UNSUPPORTED_OPERATION, 'sample() cannot be used with join()')

        # Check paramter combinations
        if (n is not None) + (n_per_stratum is not None) + (fraction is not None) != 1:
            raise excs.RequestError(
                excs.ErrorCode.MISSING_REQUIRED, 'Exactly one of `n`, `n_per_stratum`, or `fraction` must be specified.'
            )
        if n_per_stratum is not None and stratify_by is None:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, 'Must specify `stratify_by` to use `n_per_stratum`'
            )

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
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT, '`stratify_by` must be a list of scalar expressions'
                )
            for expr in stratify_by:
                if expr is None or not isinstance(expr, exprs.Expr):
                    raise excs.RequestError(excs.ErrorCode.INVALID_EXPRESSION, f'Invalid expression: {expr}')
                if not expr.col_type.is_scalar_type():
                    raise excs.RequestError(
                        excs.ErrorCode.INVALID_ARGUMENT,
                        f'Invalid type: expression must be a scalar type (not `{expr.col_type}`)',
                    )
                if not expr.is_bound_by(self._from_clause.tbls):
                    raise excs.RequestError(
                        excs.ErrorCode.UNSUPPORTED_OPERATION,
                        f"That expression cannot be evaluated in the context of this query's tables "
                        f'({",".join(tbl.tbl_name() for tbl in self._from_clause.tbls)}): {expr}',
                    )
                stratify_exprs.append(expr)

        sample_clause = SampleClause(None, n, n_per_stratum, fraction, seed, stratify_exprs)

        return type(self)(
            from_clause=self._from_clause,
            select_list=self.select_list,
            where_clause=self.where_clause,
            group_by_clause=self.group_by_clause,
            grouping_tbl_key=self.grouping_tbl_key,
            order_by_clause=self.order_by_clause,
            limit=self.limit_val,
            offset=self.offset_val,
            sample_clause=sample_clause,
        )

    def as_dict(self) -> dict[str, Any]:
        d = {
            # the concrete class, so that from_dict() can reconstruct this query as the subclass it is
            '_classname': type(self).__name__,
            'from_clause': {
                'tbls': [tbl.key().as_dict() for tbl in self._from_clause.tbls],
                'join_clauses': [
                    {
                        'join_type': clause.join_type.name,
                        'join_predicate': clause.join_predicate.as_dict()
                        if clause.join_predicate is not None
                        else None,
                    }
                    for clause in self._from_clause.join_clauses
                ],
            },
            'select_list': [(e.as_dict(), name) for (e, name) in self.select_list]
            if self.select_list is not None
            else None,
            'where_clause': self.where_clause.as_dict() if self.where_clause is not None else None,
            'group_by_clause': [e.as_dict() for e in self.group_by_clause]
            if self.group_by_clause is not None
            else None,
            'grouping_tbl': self.grouping_tbl_key.as_dict() if self.grouping_tbl_key is not None else None,
            'order_by_clause': [(e.as_dict(), asc) for (e, asc) in self.order_by_clause]
            if self.order_by_clause is not None
            else None,
            'limit_val': self.limit_val.as_dict() if self.limit_val is not None else None,
            'offset_val': self.offset_val.as_dict() if self.offset_val is not None else None,
            'sample_clause': self.sample_clause.as_dict() if self.sample_clause is not None else None,
        }
        return d

    @classmethod
    def _subclass(cls, classname: str) -> type[QueryBase]:
        # imported here because Query's module imports this one
        if classname == 'Query':
            from pixeltable._query import Query

            return Query
        if classname == 'ModelQuery':
            # ModelQuery refuses to serialize, so a stored one is a query that escaped binding
            raise excs.Error(
                excs.ErrorCode.INTERNAL_ERROR,
                'This metadata holds a query over a model, which names a table that does not exist.',
            )
        raise excs.Error(
            excs.ErrorCode.INTERNAL_ERROR,
            f'Unknown query class {classname!r}; it may have been written by a newer version of Pixeltable.',
        )

    @classmethod
    def _from_clause_from_dict(cls, d: dict[str, Any]) -> FromClause:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QueryBase:
        assert get_runtime().in_xact, 'Run from_dict() in a transaction because it may involve metadata loading'
        target = cls._subclass(d['_classname'])
        select_list = (
            [(exprs.Expr.from_dict(e), name) for e, name in d['select_list']] if d['select_list'] is not None else None
        )
        return target(
            from_clause=target._from_clause_from_dict(d['from_clause']),
            select_list=select_list,
            where_clause=exprs.Expr.from_dict(d['where_clause']) if d['where_clause'] is not None else None,
            group_by_clause=[exprs.Expr.from_dict(e) for e in d['group_by_clause']]
            if d['group_by_clause'] is not None
            else None,
            grouping_tbl_key=catalog.TableVersionKey.from_dict(d['grouping_tbl'])
            if d['grouping_tbl'] is not None
            else None,
            order_by_clause=[(exprs.Expr.from_dict(e), asc) for e, asc in d['order_by_clause']]
            if d['order_by_clause'] is not None
            else None,
            limit=exprs.Expr.from_dict(d['limit_val']) if d['limit_val'] is not None else None,
            offset=exprs.Expr.from_dict(d['offset_val']) if d.get('offset_val') is not None else None,
            sample_clause=SampleClause.from_dict(d['sample_clause']) if d['sample_clause'] is not None else None,
        )

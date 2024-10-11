import logging
import warnings
from decimal import Decimal
from typing import Optional, Iterable, Iterator, NamedTuple
from uuid import UUID

import sqlalchemy as sql

import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class OrderByItem(NamedTuple):
    expr: exprs.Expr
    asc: Optional[bool]


OrderByClause = list[OrderByItem]


def combine_order_by_clauses(clauses: Iterable[OrderByClause]) -> Optional[OrderByClause]:
    """Returns a clause that's compatible with 'clauses', or None if that doesn't exist.
    Two clauses are compatible if for each item a) the exprs are identical and b) the asc values are identical or
    at least one is None (None serves as a wildcard).
    """
    result: OrderByClause = []
    for clause in clauses:
        combined: OrderByClause = []
        for item1, item2 in zip(result, clause):
            if item1.expr.id != item2.expr.id:
                return None
            if item1.asc is not None and item2.asc is not None and item1.asc != item2.asc:
                return None
            asc = item1.asc if item1.asc is not None else item2.asc
            combined.append(OrderByItem(item1.expr, asc))

        # add remaining ordering of the longer list
        prefix_len = min(len(result), len(clause))
        if len(result) > prefix_len:
            combined.extend(result[prefix_len:])
        elif len(clause) > prefix_len:
            combined.extend(clause[prefix_len:])
        result = combined
    return result


def print_order_by_clause(clause: OrderByClause) -> str:
    return ', '.join([
        f'({item.expr}{", asc=True" if item.asc is True else ""}{", asc=False" if item.asc is False else ""})'
        for item in clause
    ])


class SqlNode(ExecNode):
    """
    Materializes data from the store via a Select stmt.
    This only provides the select list. The subclasses are responsible for the From clause and any additional clauses.
    """

    tbl: Optional[catalog.TableVersionPath]
    select_list: exprs.ExprSet
    set_pk: bool
    num_pk_cols: int
    filter: Optional[exprs.Expr]
    filter_eval_ctx: Optional[exprs.RowBuilder.EvalCtx]
    cte: Optional[sql.CTE]
    sql_elements: exprs.SqlElementCache
    limit: Optional[int]
    order_by_clause: OrderByClause

    def __init__(
            self, tbl: Optional[catalog.TableVersionPath], row_builder: exprs.RowBuilder,
            select_list: Iterable[exprs.Expr], sql_elements: exprs.SqlElementCache, set_pk: bool = False
    ):
        """
        If row_builder contains references to unstored iter columns, expands the select list to include their
        SQL-materializable subexpressions.

        Args:
            select_list: output of the query
            set_pk: if True, sets the primary for each DataRow
        """
        # create Select stmt
        self.sql_elements = sql_elements
        self.tbl = tbl
        self.select_list = exprs.ExprSet(select_list)
        # unstored iter columns: we also need to retrieve whatever is needed to materialize the iter args
        for iter_arg in row_builder.unstored_iter_args.values():
            sql_subexprs = iter_arg.subexprs(filter=self.sql_elements.contains, traverse_matches=False)
            for e in sql_subexprs:
                self.select_list.add(e)
        super().__init__(row_builder, self.select_list, [], None)  # we materialize self.select_list

        if tbl is not None:
            # minimize the number of tables that need to be joined to the target table
            self.retarget_rowid_refs(tbl, self.select_list)

        assert self.sql_elements.contains(self.select_list)
        self.set_pk = set_pk
        self.num_pk_cols = 0
        if set_pk:
            # we also need to retrieve the pk columns
            assert tbl is not None
            self.num_pk_cols = len(tbl.tbl_version.store_tbl.pk_columns())

        # additional state
        self.result_cursor = None
        # the filter is provided by the subclass
        self.filter = None
        self.filter_eval_ctx = None
        self.cte = None
        self.limit = None
        self.order_by_clause = []

    def _create_stmt(self) -> sql.Select:
        """Create Select from local state"""

        assert self.sql_elements.contains(self.select_list)
        sql_select_list = [self.sql_elements.get(e) for e in self.select_list]
        if self.set_pk:
            sql_select_list += self.tbl.tbl_version.store_tbl.pk_columns()
        stmt = sql.select(*sql_select_list)

        order_by_clause: list[sql.ClauseElement] = []
        for e, asc in self.order_by_clause:
            if isinstance(e, exprs.SimilarityExpr):
                order_by_clause.append(e.as_order_by_clause(asc))
            else:
                order_by_clause.append(self.sql_elements.get(e).desc() if asc is False else self.sql_elements.get(e))
        stmt = stmt.order_by(*order_by_clause)

        if self.filter is None and self.limit is not None:
            # if we don't have a Python filter, we can apply the limit to stmt
            stmt = stmt.limit(self.limit)

        return stmt

    def _ordering_tbl_ids(self) -> set[UUID]:
        return exprs.Expr.list_tbl_ids(e for e, _ in self.order_by_clause)

    def to_cte(self) -> Optional[tuple[sql.CTE, exprs.ExprDict[sql.ColumnElement]]]:
        """
        Returns a CTE that materializes the output of this node plus a mapping from select list expr to output column

        Returns:
            (CTE, dict from Expr to output column)
        """
        if self.filter is not None:
            # the filter needs to run in Python
            return None
        self.set_pk = False  # we don't need the PK if we use this SqlNode as a CTE
        if self.cte is None:
            self.cte = self._create_stmt().cte()
            assert len(self.cte.c) == len(self.select_list)
        return self.cte, exprs.ExprDict(zip(self.select_list, self.cte.c))

    @classmethod
    def retarget_rowid_refs(cls, target: catalog.TableVersionPath, expr_seq: Iterable[exprs.Expr]) -> None:
        """Change rowid refs to point to target"""
        for e in expr_seq:
            if isinstance(e, exprs.RowidRef):
                e.set_tbl(target)

    @classmethod
    def create_from_clause(
            cls, tbl: catalog.TableVersionPath, stmt: sql.Select, refd_tbl_ids: Optional[set[UUID]] = None,
            exact_version_only: Optional[set[UUID]] = None
    ) -> sql.Select:
        """Add From clause to stmt for tables/views referenced by materialized_exprs
        Args:
            tbl: root table of join chain
            stmt: stmt to add From clause to
            materialized_exprs: list of exprs that reference tables in the join chain; if empty, include only the root
            exact_version_only: set of table ids for which we only want to see rows created at the current version
        Returns:
            augmented stmt
        """
        # we need to include at least the root
        if refd_tbl_ids is None:
            refd_tbl_ids = {}
        if exact_version_only is None:
            exact_version_only = {}
        candidates = tbl.get_tbl_versions()
        assert len(candidates) > 0
        joined_tbls: list[catalog.TableVersion] = [candidates[0]]
        for tbl in candidates[1:]:
            if tbl.id in refd_tbl_ids:
                joined_tbls.append(tbl)

        first = True
        for tbl in joined_tbls[::-1]:
            if first:
                stmt = stmt.select_from(tbl.store_tbl.sa_tbl)
                first = False
            else:
                # join tbl to prev_tbl on prev_tbl's rowid cols
                prev_tbl_rowid_cols = prev_tbl.store_tbl.rowid_columns()
                tbl_rowid_cols = tbl.store_tbl.rowid_columns()
                rowid_clauses = \
                    [c1 == c2 for c1, c2 in zip(prev_tbl_rowid_cols, tbl_rowid_cols[:len(prev_tbl_rowid_cols)])]
                stmt = stmt.join(tbl.store_tbl.sa_tbl, sql.and_(*rowid_clauses))
            if tbl.id in exact_version_only:
                stmt = stmt.where(tbl.store_tbl.v_min_col == tbl.version)
            else:
                stmt = stmt \
                    .where(tbl.store_tbl.v_min_col <= tbl.version) \
                    .where(tbl.store_tbl.v_max_col > tbl.version)
            prev_tbl = tbl
        return stmt

    def add_order_by(self, ordering: OrderByClause) -> None:
        """Add Order By clause to stmt"""
        if self.tbl is not None:
            # change rowid refs against a base table to rowid refs against the target table, so that we minimize
            # the number of tables that need to be joined to the target table
            self.retarget_rowid_refs(self.tbl, [e for e, _ in ordering])
        combined = combine_order_by_clauses([self.order_by_clause, ordering])
        assert combined is not None
        self.order_by_clause = combined

    def set_limit(self, limit: int) -> None:
        self.limit = limit

    def _log_explain(self, stmt: sql.Select) -> None:
        try:
            # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
            stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
            explain_result = self.ctx.conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
            explain_str = '\n'.join([str(row) for row in explain_result])
            _logger.debug(f'SqlScanNode explain:\n{explain_str}')
        except Exception as e:
            _logger.warning(f'EXPLAIN failed')

    def __iter__(self) -> Iterator[DataRowBatch]:
        # run the query; do this here rather than in _open(), exceptions are only expected during iteration
        assert self.ctx.conn is not None
        try:
            with warnings.catch_warnings(record=True) as w:
                stmt = self._create_stmt()
                try:
                    # log stmt, if possible
                    stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
                    _logger.debug(f'SqlLookupNode stmt:\n{stmt_str}')
                except Exception as e:
                    pass
                self._log_explain(stmt)

                result_cursor = self.ctx.conn.execute(stmt)
                for warning in w:
                    pass
        except Exception as e:
            raise e

        tbl_version = self.tbl.tbl_version if self.tbl is not None else None
        output_batch = DataRowBatch(tbl_version, self.row_builder)
        output_row: Optional[exprs.DataRow] = None
        num_rows_returned = 0

        for sql_row in result_cursor:
            output_row = output_batch.add_row(output_row)

            # populate output_row
            if self.num_pk_cols > 0:
                output_row.set_pk(tuple(sql_row[-self.num_pk_cols:]))
            # copy the output of the SQL query into the output row
            for i, e in enumerate(self.select_list):
                slot_idx = e.slot_idx
                # certain numerical operations can produce Decimals (eg, SUM(<int column>)); we need to convert them
                if isinstance(sql_row[i], Decimal):
                    if e.col_type.is_int_type():
                        output_row[slot_idx] = int(sql_row[i])
                    elif e.col_type.is_float_type():
                        output_row[slot_idx] = float(sql_row[i])
                    else:
                        raise RuntimeError(f'Unexpected Decimal value for {e}')
                else:
                    output_row[slot_idx] = sql_row[i]

            if self.filter is not None:
                # evaluate filter
                self.row_builder.eval(output_row, self.filter_eval_ctx, profile=self.ctx.profile)
            if self.filter is not None and not output_row[self.filter.slot_idx]:
                # we re-use this row for the next sql row since it didn't pass the filter
                output_row = output_batch.pop_row()
                output_row.clear()
            else:
                # reset output_row in order to add new one
                output_row = None
                num_rows_returned += 1

            if self.limit is not None and num_rows_returned == self.limit:
                break

            if self.ctx.batch_size > 0 and len(output_batch) == self.ctx.batch_size:
                _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
                yield output_batch
                output_batch = DataRowBatch(tbl_version, self.row_builder)

        if len(output_batch) > 0:
            _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
            yield output_batch

    def _close(self) -> None:
        if self.result_cursor is not None:
            self.result_cursor.close()


class SqlScanNode(SqlNode):
    """
    Materializes data from the store via a Select stmt.

    Supports filtering and ordering.
    """
    where_clause: Optional[exprs.Expr]
    exact_version_only: list[catalog.TableVersion]

    def __init__(
            self, tbl: catalog.TableVersionPath, row_builder: exprs.RowBuilder,
            select_list: Iterable[exprs.Expr],
            where_clause: Optional[exprs.Expr] = None, filter: Optional[exprs.Expr] = None,
            set_pk: bool = False, exact_version_only: Optional[list[catalog.TableVersion]] = None
    ):
        """
        Args:
            select_list: output of the query
            sql_where_clause: SQL Where clause
            filter: additional Where-clause predicate that can't be evaluated via SQL
            limit: max number of rows to return: 0 = no limit
            set_pk: if True, sets the primary for each DataRow
            exact_version_only: tables for which we only want to see rows created at the current version
        """
        sql_elements = exprs.SqlElementCache()
        super().__init__(tbl, row_builder, select_list, sql_elements, set_pk=set_pk)
        # create Select stmt
        if exact_version_only is None:
            exact_version_only = []
        target = tbl.tbl_version  # the stored table we're scanning
        self.filter = filter
        self.filter_eval_ctx = \
            row_builder.create_eval_ctx([filter], exclude=select_list) if filter is not None else None

        self.where_clause = where_clause
        self.exact_version_only = exact_version_only

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        where_clause_tbl_ids = self.where_clause.tbl_ids() if self.where_clause is not None else set()
        refd_tbl_ids = exprs.Expr.list_tbl_ids(self.select_list) | where_clause_tbl_ids | self._ordering_tbl_ids()
        stmt = self.create_from_clause(
            self.tbl, stmt, refd_tbl_ids, exact_version_only={t.id for t in self.exact_version_only})

        if self.where_clause is not None:
            sql_where_clause = self.sql_elements.get(self.where_clause)
            assert sql_where_clause is not None
            stmt = stmt.where(sql_where_clause)

        return stmt


class SqlLookupNode(SqlNode):
    """
    Materializes data from the store via a Select stmt with a WHERE clause that matches a list of key values
    """

    where_clause: sql.ColumnElement

    def __init__(
            self, tbl: catalog.TableVersionPath, row_builder: exprs.RowBuilder,
            select_list: Iterable[exprs.Expr], sa_key_cols: list[sql.Column], key_vals: list[tuple],
    ):
        """
        Args:
            select_list: output of the query
            sa_key_cols: list of key columns in the store table
            key_vals: list of key values to look up
        """
        sql_elements = exprs.SqlElementCache()
        super().__init__(tbl, row_builder, select_list, sql_elements, set_pk=True)
        # Where clause: (key-col-1, key-col-2, ...) IN ((val-1, val-2, ...), ...)
        self.where_clause = sql.tuple_(*sa_key_cols).in_(key_vals)

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        refd_tbl_ids = exprs.Expr.list_tbl_ids(self.select_list) | self._ordering_tbl_ids()
        stmt = self.create_from_clause(self.tbl, stmt, refd_tbl_ids)
        stmt = stmt.where(self.where_clause)
        return stmt

class SqlAggregationNode(SqlNode):
    """
    Materializes data from the store via a Select stmt with a WHERE clause that matches a list of key values
    """

    group_by_items: Optional[list[exprs.Expr]]

    def __init__(
            self, row_builder: exprs.RowBuilder,
            input: SqlNode,
            select_list: Iterable[exprs.Expr],
            group_by_items: Optional[list[exprs.Expr]] = None,
            limit: Optional[int] = None, exact_version_only: Optional[list[catalog.TableVersion]] = None
    ):
        """
        Args:
            select_list: can contain calls to AggregateFunctions
            group_by_items: list of expressions to group by
            limit: max number of rows to return: None = no limit
        """
        _, input_col_map = input.to_cte()
        sql_elements = exprs.SqlElementCache(input_col_map)
        super().__init__(None, row_builder, select_list, sql_elements)
        self.group_by_items = group_by_items

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        if self.group_by_items is not None:
            sql_group_by_items = [self.sql_elements.get(e) for e in self.group_by_items]
            assert all(e is not None for e in sql_group_by_items)
            stmt = stmt.group_by(*sql_group_by_items)
        return stmt

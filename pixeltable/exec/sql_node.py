import logging
import warnings
from decimal import Decimal
from typing import TYPE_CHECKING, AsyncIterator, Iterable, NamedTuple, Optional, Sequence
from uuid import UUID

import sqlalchemy as sql

from pixeltable import catalog, exprs
from pixeltable.env import Env

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

if TYPE_CHECKING:
    import pixeltable.plan
    from pixeltable.plan import SampleClause

_logger = logging.getLogger('pixeltable')


class OrderByItem(NamedTuple):
    expr: exprs.Expr
    asc: Optional[bool]


OrderByClause = list[OrderByItem]


def combine_order_by_clauses(clauses: Iterable[OrderByClause]) -> Optional[OrderByClause]:
    """Returns a clause that's compatible with 'clauses', or None if that doesn't exist.
    Two clauses are compatible if for each of their respective items c1[i] and c2[i]
    a) the exprs are identical and
    b) the asc values are identical or at least one is None (None serves as a wildcard)
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
    return ', '.join(
        [
            f'({item.expr}{", asc=True" if item.asc is True else ""}{", asc=False" if item.asc is False else ""})'
            for item in clause
        ]
    )


class SqlNode(ExecNode):
    """
    Materializes data from the store via an SQL statement.
    This only provides the select list. The subclasses are responsible for the From clause and any additional clauses.
    The pk columns are not included in the select list.
    If set_pk is True, they are added to the end of the result set when creating the SQL statement
    so they can always be referenced as cols[-num_pk_cols:] in the result set.
    The pk_columns consist of the rowid columns of the target table followed by the version number.
    """

    tbl: Optional[catalog.TableVersionPath]
    select_list: exprs.ExprSet
    set_pk: bool
    num_pk_cols: int
    py_filter: Optional[exprs.Expr]  # a predicate that can only be run in Python
    py_filter_eval_ctx: Optional[exprs.RowBuilder.EvalCtx]
    cte: Optional[sql.CTE]
    sql_elements: exprs.SqlElementCache

    # where_clause/-_element: allow subclass to set one or the other (but not both)
    where_clause: Optional[exprs.Expr]
    where_clause_element: Optional[sql.ColumnElement]

    order_by_clause: OrderByClause
    limit: Optional[int]

    def __init__(
        self,
        tbl: Optional[catalog.TableVersionPath],
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        sql_elements: exprs.SqlElementCache,
        set_pk: bool = False,
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

        assert self.sql_elements.contains_all(self.select_list)
        self.set_pk = set_pk
        self.num_pk_cols = 0
        if set_pk:
            # we also need to retrieve the pk columns
            assert tbl is not None
            self.num_pk_cols = len(tbl.tbl_version.get().store_tbl.pk_columns())
            assert self.num_pk_cols > 1

        # additional state
        self.result_cursor = None
        # the filter is provided by the subclass
        self.py_filter = None
        self.py_filter_eval_ctx = None
        self.cte = None
        self.limit = None
        self.where_clause = None
        self.where_clause_element = None
        self.order_by_clause = []

        if self.tbl is not None:
            tv = self.tbl.tbl_version._tbl_version
            if tv is not None:
                assert tv.is_validated

    def _create_pk_cols(self) -> list[sql.Column]:
        """Create a list of pk columns"""
        # we need to retrieve the pk columns
        if self.set_pk:
            assert self.tbl is not None
            assert self.tbl.tbl_version.get().is_validated
            return self.tbl.tbl_version.get().store_tbl.pk_columns()
        return []

    def _create_stmt(self) -> sql.Select:
        """Create Select from local state"""

        assert self.sql_elements.contains_all(self.select_list)
        sql_select_list = [self.sql_elements.get(e) for e in self.select_list] + self._create_pk_cols()
        stmt = sql.select(*sql_select_list)

        where_clause_element = (
            self.sql_elements.get(self.where_clause) if self.where_clause is not None else self.where_clause_element
        )
        if where_clause_element is not None:
            stmt = stmt.where(where_clause_element)

        order_by_clause: list[sql.ColumnElement] = []
        for e, asc in self.order_by_clause:
            if isinstance(e, exprs.SimilarityExpr):
                order_by_clause.append(e.as_order_by_clause(asc))
            else:
                order_by_clause.append(self.sql_elements.get(e).desc() if asc is False else self.sql_elements.get(e))
        stmt = stmt.order_by(*order_by_clause)

        if self.py_filter is None and self.limit is not None:
            # if we don't have a Python filter, we can apply the limit to stmt
            stmt = stmt.limit(self.limit)

        return stmt

    def _ordering_tbl_ids(self) -> set[UUID]:
        return exprs.Expr.all_tbl_ids(e for e, _ in self.order_by_clause)

    def to_cte(self, keep_pk: bool = False) -> Optional[tuple[sql.CTE, exprs.ExprDict[sql.ColumnElement]]]:
        """
        Creates a CTE that materializes the output of this node plus a mapping from select list expr to output column.
        keep_pk: if True, the PK columns are included in the CTE Select statement

        Returns:
            (CTE, dict from Expr to output column)
        """
        if self.py_filter is not None:
            # the filter needs to run in Python
            return None
        if self.cte is None:
            if not keep_pk:
                self.set_pk = False  # we don't need the PK if we use this SqlNode as a CTE
            self.cte = self._create_stmt().cte()
        pk_count = self.num_pk_cols if self.set_pk else 0
        assert len(self.select_list) + pk_count == len(self.cte.c)
        return self.cte, exprs.ExprDict(zip(self.select_list, self.cte.c))  # skip pk cols

    @classmethod
    def retarget_rowid_refs(cls, target: catalog.TableVersionPath, expr_seq: Iterable[exprs.Expr]) -> None:
        """Change rowid refs to point to target"""
        for e in expr_seq:
            if isinstance(e, exprs.RowidRef):
                e.set_tbl(target)

    @classmethod
    def create_from_clause(
        cls,
        tbl: catalog.TableVersionPath,
        stmt: sql.Select,
        refd_tbl_ids: Optional[set[UUID]] = None,
        exact_version_only: Optional[set[UUID]] = None,
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
            refd_tbl_ids = set()
        if exact_version_only is None:
            exact_version_only = set()
        candidates = tbl.get_tbl_versions()
        assert len(candidates) > 0
        joined_tbls: list[catalog.TableVersionHandle] = [candidates[0]]
        for t in candidates[1:]:
            if t.id in refd_tbl_ids:
                joined_tbls.append(t)

        first = True
        prev_tv: Optional[catalog.TableVersion] = None
        for t in joined_tbls[::-1]:
            tv = t.get()
            # _logger.debug(f'create_from_clause: tbl_id={tv.id} {id(tv.store_tbl.sa_tbl)}')
            if first:
                stmt = stmt.select_from(tv.store_tbl.sa_tbl)
                first = False
            else:
                # join tv to prev_tv on prev_tv's rowid cols
                prev_tbl_rowid_cols = prev_tv.store_tbl.rowid_columns()
                tbl_rowid_cols = tv.store_tbl.rowid_columns()
                rowid_clauses = [
                    c1 == c2 for c1, c2 in zip(prev_tbl_rowid_cols, tbl_rowid_cols[: len(prev_tbl_rowid_cols)])
                ]
                stmt = stmt.join(tv.store_tbl.sa_tbl, sql.and_(*rowid_clauses))

            if t.id in exact_version_only:
                stmt = stmt.where(tv.store_tbl.v_min_col == tv.version)
            else:
                stmt = stmt.where(tv.store_tbl.sa_tbl.c.v_min <= tv.version)
                stmt = stmt.where(tv.store_tbl.sa_tbl.c.v_max > tv.version)
            prev_tv = tv

        return stmt

    def set_where(self, where_clause: exprs.Expr) -> None:
        assert self.where_clause_element is None
        self.where_clause = where_clause

    def set_py_filter(self, py_filter: exprs.Expr) -> None:
        assert self.py_filter is None
        self.py_filter = py_filter
        self.py_filter_eval_ctx = self.row_builder.create_eval_ctx([py_filter], exclude=self.select_list)

    def set_order_by(self, ordering: OrderByClause) -> None:
        """Add Order By clause"""
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
        conn = Env.get().conn
        try:
            # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
            stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
            explain_result = conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
            explain_str = '\n'.join([str(row) for row in explain_result])
            _logger.debug(f'SqlScanNode explain:\n{explain_str}')
        except Exception as e:
            _logger.warning(f'EXPLAIN failed with error: {e}')

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        # run the query; do this here rather than in _open(), exceptions are only expected during iteration
        with warnings.catch_warnings(record=True) as w:
            stmt = self._create_stmt()
            try:
                # log stmt, if possible
                stmt_str = str(stmt.compile(compile_kwargs={'literal_binds': True}))
                _logger.debug(f'SqlLookupNode stmt:\n{stmt_str}')
            except Exception:
                # log something if we can't log the compiled stmt
                _logger.debug(f'SqlLookupNode proto-stmt:\n{stmt}')
            self._log_explain(stmt)

            conn = Env.get().conn
            result_cursor = conn.execute(stmt)
            for _ in w:
                pass

        tbl_version = self.tbl.tbl_version if self.tbl is not None else None
        output_batch = DataRowBatch(tbl_version, self.row_builder)
        output_row: Optional[exprs.DataRow] = None
        num_rows_returned = 0

        for sql_row in result_cursor:
            output_row = output_batch.add_row(output_row)

            # populate output_row
            if self.num_pk_cols > 0:
                output_row.set_pk(tuple(sql_row[-self.num_pk_cols :]))
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

            if self.py_filter is not None:
                # evaluate filter
                self.row_builder.eval(output_row, self.py_filter_eval_ctx, profile=self.ctx.profile)
            if self.py_filter is not None and not output_row[self.py_filter.slot_idx]:
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

    exact_version_only: list[catalog.TableVersionHandle]

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        set_pk: bool = False,
        exact_version_only: Optional[list[catalog.TableVersionHandle]] = None,
    ):
        """
        Args:
            select_list: output of the query
            set_pk: if True, sets the primary for each DataRow
            exact_version_only: tables for which we only want to see rows created at the current version
        """
        sql_elements = exprs.SqlElementCache()
        super().__init__(tbl, row_builder, select_list, sql_elements, set_pk=set_pk)
        # create Select stmt
        if exact_version_only is None:
            exact_version_only = []

        self.exact_version_only = exact_version_only

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        where_clause_tbl_ids = self.where_clause.tbl_ids() if self.where_clause is not None else set()
        refd_tbl_ids = exprs.Expr.all_tbl_ids(self.select_list) | where_clause_tbl_ids | self._ordering_tbl_ids()
        stmt = self.create_from_clause(
            self.tbl, stmt, refd_tbl_ids, exact_version_only={t.id for t in self.exact_version_only}
        )
        return stmt


class SqlLookupNode(SqlNode):
    """
    Materializes data from the store via a Select stmt with a WHERE clause that matches a list of key values
    """

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        row_builder: exprs.RowBuilder,
        select_list: Iterable[exprs.Expr],
        sa_key_cols: list[sql.Column],
        key_vals: list[tuple],
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
        self.where_clause_element = sql.tuple_(*sa_key_cols).in_(key_vals)

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt()
        refd_tbl_ids = exprs.Expr.all_tbl_ids(self.select_list) | self._ordering_tbl_ids()
        stmt = self.create_from_clause(self.tbl, stmt, refd_tbl_ids)
        return stmt


class SqlAggregationNode(SqlNode):
    """
    Materializes data from the store via a Select stmt with a WHERE clause that matches a list of key values
    """

    group_by_items: Optional[list[exprs.Expr]]
    input_cte: Optional[sql.CTE]

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        input: SqlNode,
        select_list: Iterable[exprs.Expr],
        group_by_items: Optional[list[exprs.Expr]] = None,
        limit: Optional[int] = None,
        exact_version_only: Optional[list[catalog.TableVersion]] = None,
    ):
        """
        Args:
            select_list: can contain calls to AggregateFunctions
            group_by_items: list of expressions to group by
            limit: max number of rows to return: None = no limit
        """
        self.input_cte, input_col_map = input.to_cte()
        sql_elements = exprs.SqlElementCache(input_col_map)
        super().__init__(None, row_builder, select_list, sql_elements)
        self.group_by_items = group_by_items

    def _create_stmt(self) -> sql.Select:
        stmt = super()._create_stmt().select_from(self.input_cte)
        if self.group_by_items is not None:
            sql_group_by_items = [self.sql_elements.get(e) for e in self.group_by_items]
            assert all(e is not None for e in sql_group_by_items)
            stmt = stmt.group_by(*sql_group_by_items)
        return stmt


class SqlJoinNode(SqlNode):
    """
    Materializes data from the store via a Select ... From ... that contains joins
    """

    input_ctes: list[sql.CTE]
    join_clauses: list['pixeltable.plan.JoinClause']

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        inputs: Sequence[SqlNode],
        join_clauses: list['pixeltable.plan.JoinClause'],
        select_list: Iterable[exprs.Expr],
    ):
        assert len(inputs) > 1
        assert len(inputs) == len(join_clauses) + 1
        self.input_ctes = []
        self.join_clauses = join_clauses
        sql_elements = exprs.SqlElementCache()
        for input_node in inputs:
            input_cte, input_col_map = input_node.to_cte()
            self.input_ctes.append(input_cte)
            sql_elements.extend(input_col_map)
        super().__init__(None, row_builder, select_list, sql_elements)

    def _create_stmt(self) -> sql.Select:
        from pixeltable import plan

        stmt = super()._create_stmt()
        stmt = stmt.select_from(self.input_ctes[0])
        for i in range(len(self.join_clauses)):
            join_clause = self.join_clauses[i]
            on_clause = (
                self.sql_elements.get(join_clause.join_predicate)
                if join_clause.join_type != plan.JoinType.CROSS
                else sql.sql.expression.literal(True)
            )
            is_outer = join_clause.join_type in (plan.JoinType.LEFT, plan.JoinType.FULL_OUTER)
            stmt = stmt.join(
                self.input_ctes[i + 1],
                onclause=on_clause,
                isouter=is_outer,
                full=join_clause == plan.JoinType.FULL_OUTER,
            )
        return stmt


class SqlSampleNode(SqlNode):
    """
    Returns rows sampled from the input node.
    """

    input_cte: Optional[sql.CTE]
    pk_count: int
    stratify_exprs: Optional[list[exprs.Expr]]
    sample_clause: 'SampleClause'

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        input: SqlNode,
        select_list: Iterable[exprs.Expr],
        sample_clause: 'SampleClause',
        stratify_exprs: list[exprs.Expr],
    ):
        """
        Args:
            input: SqlNode to sample from
            select_list: can contain calls to AggregateFunctions
            sample_clause: specifies the sampling method
            stratify_exprs: Analyzer processed list of expressions to stratify by.
        """
        assert isinstance(input, SqlNode)
        self.input_cte, input_col_map = input.to_cte(keep_pk=True)
        self.pk_count = input.num_pk_cols
        assert self.pk_count > 1
        sql_elements = exprs.SqlElementCache(input_col_map)
        assert sql_elements.contains_all(stratify_exprs)
        super().__init__(input.tbl, row_builder, select_list, sql_elements, set_pk=True)
        self.stratify_exprs = stratify_exprs
        self.sample_clause = sample_clause
        assert isinstance(self.sample_clause.seed, int)

    @classmethod
    def key_sql_expr(cls, seed: sql.ColumnElement, sql_cols: Iterable[sql.ColumnElement]) -> sql.ColumnElement:
        """Construct expression which is the ordering key for rows to be sampled
        General SQL form is:
        - MD5(<seed::text> [ + '___' + <rowid_col_val>::text]+
        """
        sql_expr: sql.ColumnElement = sql.cast(seed, sql.Text)
        for e in sql_cols:
            # Quotes are required below to guarantee that the string is properly presented in SQL
            sql_expr = sql_expr + sql.literal_column("'___'", sql.Text) + sql.cast(e, sql.Text)
        sql_expr = sql.func.md5(sql_expr)
        return sql_expr

    def _create_key_sql(self, cte: sql.CTE) -> sql.ColumnElement:
        """Create an expression for randomly ordering rows with a given seed"""
        rowid_cols = [*cte.c[-self.pk_count : -1]]  # exclude the version column
        assert len(rowid_cols) > 0
        return self.key_sql_expr(sql.literal_column(str(self.sample_clause.seed)), rowid_cols)

    def _create_stmt(self) -> sql.Select:
        from pixeltable.plan import SampleClause

        if self.sample_clause.fraction is not None:
            if len(self.stratify_exprs) == 0:
                # If non-stratified sampling, construct a where clause, order_by, and limit clauses
                s_key = self._create_key_sql(self.input_cte)

                # Construct a suitable where clause
                fraction_sql = sql.cast(SampleClause.fraction_to_md5_hex(float(self.sample_clause.fraction)), sql.Text)
                order_by = self._create_key_sql(self.input_cte)
                return sql.select(*self.input_cte.c).where(s_key < fraction_sql).order_by(order_by)

            return self._create_stmt_stratified_fraction(self.sample_clause.fraction)
        else:
            if len(self.stratify_exprs) == 0:
                # No stratification, just return n samples from the input CTE
                order_by = self._create_key_sql(self.input_cte)
                return sql.select(*self.input_cte.c).order_by(order_by).limit(self.sample_clause.n)

            return self._create_stmt_stratified_n(self.sample_clause.n, self.sample_clause.n_per_stratum)

    def _create_stmt_stratified_n(self, n: Optional[int], n_per_stratum: Optional[int]) -> sql.Select:
        """Create a Select stmt that returns n samples across all strata or n_per_stratum samples per stratum"""

        sql_strata_exprs = [self.sql_elements.get(e) for e in self.stratify_exprs]
        order_by = self._create_key_sql(self.input_cte)

        # Create a list of all columns plus the rank
        # Get all columns from the input CTE dynamically
        select_columns = [*self.input_cte.c]
        select_columns.append(
            sql.func.row_number().over(partition_by=sql_strata_exprs, order_by=order_by).label('rank')
        )
        row_rank_cte = sql.select(*select_columns).select_from(self.input_cte).cte('row_rank_cte')

        final_columns = [*row_rank_cte.c[:-1]]  # exclude the rank column
        if n_per_stratum is not None:
            return sql.select(*final_columns).filter(row_rank_cte.c.rank <= n_per_stratum)
        else:
            secondary_order = self._create_key_sql(row_rank_cte)
            return sql.select(*final_columns).order_by(row_rank_cte.c.rank, secondary_order).limit(n)

    def _create_stmt_stratified_fraction(self, fraction_samples: float) -> sql.Select:
        """Create a Select stmt that returns a fraction of the rows per strata"""

        # Build the strata count CTE
        # Produces a table of the form:
        #   (*stratify_exprs, s_s_size)
        # where s_s_size is the number of samples to take from each stratum
        sql_strata_exprs = [self.sql_elements.get(e) for e in self.stratify_exprs]
        per_strata_count_cte = (
            sql.select(
                *sql_strata_exprs,
                sql.func.ceil(fraction_samples * sql.func.count(1).cast(sql.Integer)).label('s_s_size'),
            )
            .select_from(self.input_cte)
            .group_by(*sql_strata_exprs)
            .cte('per_strata_count_cte')
        )

        # Build a CTE that ranks the rows within each stratum
        # Include all columns from the input CTE dynamically
        order_by = self._create_key_sql(self.input_cte)
        select_columns = [*self.input_cte.c]
        select_columns.append(
            sql.func.row_number().over(partition_by=sql_strata_exprs, order_by=order_by).label('rank')
        )
        row_rank_cte = sql.select(*select_columns).select_from(self.input_cte).cte('row_rank_cte')

        # Build the join criterion dynamically to accommodate any number of stratify_by expressions
        join_c = sql.true()
        for col in per_strata_count_cte.c[:-1]:
            join_c &= row_rank_cte.c[col.name].isnot_distinct_from(col)

        # Join with per_strata_count_cte to limit returns to the requested fraction of rows
        final_columns = [*row_rank_cte.c[:-1]]  # exclude the rank column
        stmt = (
            sql.select(*final_columns)
            .select_from(row_rank_cte)
            .join(per_strata_count_cte, join_c)
            .where(row_rank_cte.c.rank <= per_strata_count_cte.c.s_s_size)
        )

        return stmt

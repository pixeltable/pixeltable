from typing import List, Optional, Tuple, Iterable, Set
from uuid import UUID
import logging
import warnings

import sqlalchemy as sql

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.exprs as exprs
import pixeltable.catalog as catalog


_logger = logging.getLogger('pixeltable')

class SqlScanNode(ExecNode):
    """Materializes data from the store via SQL
    """
    def __init__(
            self, tbl: catalog.TableVersionPath, row_builder: exprs.RowBuilder,
            select_list: Iterable[exprs.Expr],
            where_clause: Optional[exprs.Expr] = None, filter: Optional[exprs.Predicate] = None,
            order_by_items: Optional[List[Tuple[exprs.Expr, bool]]] = None,
            similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None,
            limit: int = 0, set_pk: bool = False, exact_version_only: Optional[List[catalog.TableVersion]] = None
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
        # create Select stmt
        if order_by_items is None:
            order_by_items = []
        if exact_version_only is None:
            exact_version_only = []
        super().__init__(row_builder, [], [], None)
        self.tbl = tbl
        target = tbl.tbl_version  # the stored table we're scanning
        self.sql_exprs = exprs.ExprSet(select_list)
        # unstored iter columns: we also need to retrieve whatever is needed to materialize the iter args
        for iter_arg in row_builder.unstored_iter_args.values():
            sql_subexprs = iter_arg.subexprs(filter=lambda e: e.sql_expr() is not None, traverse_matches=False)
            [self.sql_exprs.append(e) for e in sql_subexprs]
        self.filter = filter
        self.filter_eval_ctx = \
            row_builder.create_eval_ctx([filter], exclude=select_list) if filter is not None else None
        self.limit = limit

        # change rowid refs against a base table to rowid refs against the target table, so that we minimize
        # the number of tables that need to be joined to the target table
        for rowid_ref in [e for e in self.sql_exprs if isinstance(e, exprs.RowidRef)]:
            rowid_ref.set_tbl(tbl)

        where_clause_tbl_ids = where_clause.tbl_ids() if where_clause is not None else set()
        refd_tbl_ids = exprs.Expr.list_tbl_ids(self.sql_exprs) | where_clause_tbl_ids
        sql_select_list = [e.sql_expr() for e in self.sql_exprs]
        assert len(sql_select_list) == len(self.sql_exprs)
        assert all([e is not None for e in sql_select_list])
        self.set_pk = set_pk
        self.num_pk_cols = 0
        if set_pk:
            # we also need to retrieve the pk columns
            pk_columns = target.store_tbl.pk_columns()
            self.num_pk_cols = len(pk_columns)
            sql_select_list += pk_columns

        self.stmt = sql.select(*sql_select_list)
        self.stmt = self.create_from_clause(
            tbl, self.stmt, refd_tbl_ids, exact_version_only={t.id for t in exact_version_only})

        # change rowid refs against a base table to rowid refs against the target table, so that we minimize
        # the number of tables that need to be joined to the target table
        for rowid_ref in [e for e, _ in order_by_items if isinstance(e, exprs.RowidRef)]:
            rowid_ref.set_tbl(tbl)
        order_by_clause = [e.sql_expr().desc() if not asc else e.sql_expr() for e, asc in order_by_items]

        if where_clause is not None:
            sql_where_clause = where_clause.sql_expr()
            assert sql_where_clause is not None
            self.stmt = self.stmt.where(sql_where_clause)
        if similarity_clause is not None:
            self.stmt = self.stmt.order_by(
                similarity_clause.img_col_ref.col.sa_idx_col.l2_distance(similarity_clause.embedding()))
        if len(order_by_clause) > 0:
            self.stmt = self.stmt.order_by(*order_by_clause)
        elif target.id in row_builder.unstored_iter_args:
            # we are referencing unstored iter columns from this view and try to order by our primary key,
            # which ensures that iterators will see monotonically increasing pos values
            self.stmt = self.stmt.order_by(*self.tbl.store_tbl.rowid_columns())
        if limit != 0 and self.filter is None:
            # if we need to do post-SQL filtering, we can't use LIMIT
            self.stmt = self.stmt.limit(limit)

        self.result_cursor: Optional[sql.engine.CursorResult] = None

        try:
            # log stmt, if possible
            stmt_str = str(self.stmt.compile(compile_kwargs={'literal_binds': True}))
            _logger.debug(f'SqlScanNode stmt:\n{stmt_str}')
        except Exception as e:
            pass

    @classmethod
    def create_from_clause(
            cls, tbl: catalog.TableVersionPath, stmt: sql.Select, refd_tbl_ids: Optional[Set[UUID]] = None,
            exact_version_only: Optional[Set[UUID]] = None
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
        joined_tbls: List[catalog.TableVersion] = [candidates[0]]
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

    def _log_explain(self, conn: sql.engine.Connection) -> None:
        try:
            # don't set dialect=Env.get().engine.dialect: x % y turns into x %% y, which results in a syntax error
            stmt_str = str(self.stmt.compile(compile_kwargs={'literal_binds': True}))
            explain_result = self.ctx.conn.execute(sql.text(f'EXPLAIN {stmt_str}'))
            explain_str = '\n'.join([str(row) for row in explain_result])
            _logger.debug(f'SqlScanNode explain:\n{explain_str}')
        except Exception as e:
            _logger.warning(f'EXPLAIN failed')

    def __next__(self) -> DataRowBatch:
        if self.result_cursor is None:
            # run the query; do this here rather than in _open(), exceptions are only expected during iteration
            assert self.ctx.conn is not None
            try:
                self._log_explain(self.ctx.conn)
                with warnings.catch_warnings(record=True) as w:
                    self.result_cursor = self.ctx.conn.execute(self.stmt)
                    for warning in w:
                        pass
                self.has_more_rows = True
            except Exception as e:
                self.has_more_rows = False
                raise e

        if not self.has_more_rows:
            raise StopIteration

        output_batch = DataRowBatch(self.tbl.tbl_version, self.row_builder)
        needs_row = True
        while self.ctx.batch_size == 0 or len(output_batch) < self.ctx.batch_size:
            try:
                sql_row = next(self.result_cursor)
            except StopIteration:
                self.has_more_rows = False
                break

            if needs_row:
                output_row = output_batch.add_row()
            if self.num_pk_cols > 0:
                output_row.set_pk(tuple(sql_row[-self.num_pk_cols:]))
            # copy the output of the SQL query into the output row
            for i, e in enumerate(self.sql_exprs):
                slot_idx = e.slot_idx
                output_row[slot_idx] = sql_row[i]
            if self.filter is not None:
                self.row_builder.eval(output_row, self.filter_eval_ctx, profile=self.ctx.profile)
                if output_row[self.filter.slot_idx]:
                    needs_row = True
                    if self.limit is not None and len(output_batch) >= self.limit:
                        self.has_more_rows = False
                        break
                else:
                    # we re-use this row for the next sql row if it didn't pass the filter
                    needs_row = False
                    output_row.clear()

        if not needs_row:
            # the last row didn't pass the filter
            assert self.filter is not None
            output_batch.pop_row()

        _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
        if len(output_batch) == 0:
            raise StopIteration
        return output_batch

    def _close(self) -> None:
        if self.result_cursor is not None:
            self.result_cursor.close()


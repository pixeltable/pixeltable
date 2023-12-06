from __future__ import annotations

from typing import List, Iterator, Set, Dict, Any, Optional, Tuple, Iterable, Generator
from dataclasses import dataclass, field
import logging
import time
import abc
import io
import sys
import urllib.parse
import urllib.request
from uuid import UUID
import concurrent.futures
import threading
import os
from collections import defaultdict
import warnings

import numpy as np
from tqdm.autonotebook import tqdm
import nos
import sqlalchemy as sql

import pixeltable.exprs as exprs
import pixeltable.catalog as catalog
from pixeltable.utils.imgstore import ImageStore
from pixeltable.function import Function, FunctionRegistry
from pixeltable.env import Env
from pixeltable import exceptions as exc
from pixeltable.utils.filecache import FileCache


_logger = logging.getLogger('pixeltable')


class DataRowBatch:
    """Set of DataRows, indexed by rowid.

    Contains the metadata needed to initialize DataRows.
    """
    def __init__(self, table: catalog.TableVersion, row_builder: exprs.RowBuilder, len: int = 0):
        self.table_id = table.id
        self.table_version = table.version
        self.row_builder = row_builder
        self.img_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_image_type()]
        self.video_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_video_type()]
        self.array_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_array_type()]
        self.rows = [
            exprs.DataRow(row_builder.num_materialized, self.img_slot_idxs, self.video_slot_idxs, self.array_slot_idxs)
            for _ in range(len)
        ]

    def add_row(self, row: Optional[exprs.DataRow] = None) -> exprs.DataRow:
        if row is None:
            row = exprs.DataRow(
                self.row_builder.num_materialized, self.img_slot_idxs, self.video_slot_idxs, self.array_slot_idxs)
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        return self.rows.pop()

    def set_row_ids(self, row_ids: List[int]) -> None:
        """Sets pks for rows in batch"""
        assert len(row_ids) == len(self.rows)
        for row, row_id in zip(self.rows, row_ids):
            row.set_pk((row_id, self.table_version))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: object) -> exprs.DataRow:
        return self.rows[index]

    def flush_imgs(
            self, idx_range: Optional[slice] = None, stored_img_info: List[exprs.ColumnSlotIdx] = [],
            flushed_slot_idxs: List[int] = []
    ) -> None:
        """Flushes images in the given range of rows."""
        if len(stored_img_info) == 0 and len(flushed_slot_idxs) == 0:
            return
        if idx_range is None:
            idx_range = slice(0, len(self.rows))
        for row in self.rows[idx_range]:
            for info in stored_img_info:
                filepath = str(ImageStore.get_path(self.table_id, info.col.id, self.table_version))
                row.flush_img(info.slot_idx, filepath)
            for slot_idx in flushed_slot_idxs:
                row.flush_img(slot_idx)
        #_logger.debug(
            #f'flushed images in range {idx_range}: slot_idxs={flushed_slot_idxs} stored_img_info={stored_img_info}')

    def __iter__(self) -> Iterator[exprs.DataRow]:
        return DataRowBatchIterator(self)


class DataRowBatchIterator:
    """
    Iterator over a DataRowBatch.
    """
    def __init__(self, batch: DataRowBatch):
        self.row_batch = batch
        self.index = 0

    def __next__(self) -> exprs.DataRow:
        if self.index >= len(self.row_batch.rows):
            raise StopIteration
        row = self.row_batch.rows[self.index]
        self.index += 1
        return row


class ExecContext:
    """Class for execution runtime constants"""
    def __init__(
            self, row_builder: exprs.RowBuilder, *, show_pbar: bool = False, batch_size: int = 0,
            pk_clause: Optional[List[sql.ClauseElement]] = None, num_computed_exprs: int = 0
    ):
        self.show_pbar = show_pbar
        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(row_builder)
        # num_rows is used to compute the total number of computed cells used for the progress bar
        self.num_rows: Optional[int] = None
        self.conn: Optional[sql.engine.Connection] = None  # if present, use this to execute SQL queries
        self.pk_clause = pk_clause
        self.num_computed_exprs = num_computed_exprs

    def set_pk_clause(self, pk_clause: List[sql.ClauseElement]) -> None:
        self.pk_clause = pk_clause


class ExecNode(abc.ABC):
    """Base class of all execution nodes"""
    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr],
            input_exprs: Iterable[exprs.Expr], input: Optional[ExecNode] = None):
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies
            if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.stored_img_cols: List[exprs.ColumnSlotIdx] = []
        self.ctx: Optional[ExecContext] = None  # all nodes of a tree share the same context

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def set_stored_img_cols(self, stored_img_cols: List[exprs.ColumnSlotIdx]) -> None:
        self.stored_img_cols = stored_img_cols
        # propagate batch size to the source
        if self.input is not None:
            self.input.set_stored_img_cols(stored_img_cols)

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self) -> DataRowBatch:
        pass

    def open(self) -> None:
        """Bottom-up initialization of nodes for execution. Must be called before __next__."""
        if self.input is not None:
            self.input.open()
        self._open()

    def close(self) -> None:
        """Frees node resources top-down after execution. Must be called after final __next__."""
        self._close()
        if self.input is not None:
            self.input.close()

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass


class AggregationNode(ExecNode):
    def __init__(
            self, tbl: catalog.TableVersion, row_builder: exprs.RowBuilder, group_by: List[exprs.Expr],
            agg_fn_calls: List[exprs.FunctionCall], input_exprs: List[exprs.Expr], input: ExecNode
    ):
        super().__init__(row_builder, group_by + agg_fn_calls, input_exprs, input)
        self.input = input
        self.group_by = group_by
        self.input_exprs = input_exprs
        self.agg_fn_calls = agg_fn_calls
        self.agg_fn_eval_ctx = row_builder.create_eval_ctx(agg_fn_calls, exclude=input_exprs)
        self.output_batch = DataRowBatch(tbl, row_builder, 0)

    def _reset_agg_state(self, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.reset_agg()
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'init() function of the aggregate {fn_call}'
                raise exc.ExprEvalError(fn_call, expr_msg, e, exc_tb, [], row_num)

    def _update_agg_state(self, row: exprs.DataRow, row_num: int) -> None:
        for fn_call in self.agg_fn_calls:
            try:
                fn_call.update(row)
            except Exception as e:
                _, _, exc_tb = sys.exc_info()
                expr_msg = f'update() function of the aggregate {fn_call}'
                input_vals = [row[d.slot_idx] for d in fn_call.dependencies()]
                raise exc.ExprEvalError(fn_call, expr_msg, e, exc_tb, input_vals, row_num)

    def __next__(self) -> DataRowBatch:
        if self.output_batch is None:
            raise StopIteration

        prev_row: Optional[exprs.DataRow] = None
        current_group: Optional[List[Any]] = None  # the values of the group-by exprs
        num_input_rows = 0
        for row_batch in self.input:
            num_input_rows += len(row_batch)
            for row in row_batch:
                group = [row[e.slot_idx] for e in self.group_by]
                if current_group is None:
                    current_group = group
                    self._reset_agg_state(0)
                if group != current_group:
                    # we're entering a new group, emit a row for the previous one
                    self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
                    self.output_batch.add_row(prev_row)
                    current_group = group
                    self._reset_agg_state(0)
                self._update_agg_state(row, 0)
                prev_row = row
        # emit the last group
        self.row_builder.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
        self.output_batch.add_row(prev_row)

        result = self.output_batch
        result.flush_imgs(None, self.stored_img_cols, self.flushed_img_slots)
        self.output_batch = None
        _logger.debug(f'AggregateNode: consumed {num_input_rows} rows, returning {len(result.rows)} rows')
        return result


class SqlScanNode(ExecNode):
    """Materializes data from the store via SQL
    """
    def __init__(
            self, tbl: catalog.TableVersion, row_builder: exprs.RowBuilder,
            select_list: Iterable[exprs.Expr],
            where_clause: Optional[exprs.Expr] = None, filter: Optional[exprs.Predicate] = None,
            order_by_items: List[Tuple[exprs.Expr, bool]] = [],
            similarity_clause: Optional[exprs.ImageSimilarityPredicate] = None,
            limit: int = 0, set_pk: bool = False, exact_version_only: List[catalog.TableVersion] = []
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
        super().__init__(row_builder, [], [], None)
        self.tbl = tbl
        self.sql_exprs = exprs.ExprSet(select_list)
        # unstored iter columns: we also need to retrieve whatever is needed to materialize the iter args
        for iter_arg in row_builder.unstored_iter_args.values():
            sql_subexprs = iter_arg.subexprs(filter=lambda e: e.sql_expr() is not None, traverse_matches=False)
            [self.sql_exprs.append(e) for e in sql_subexprs]
        self.filter = filter
        self.filter_eval_ctx = row_builder.create_eval_ctx([filter], exclude=select_list) if filter is not None else []
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
        self.num_pk_cols = 0  # set in _open()

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
        elif tbl.id in row_builder.unstored_iter_args:
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
            cls, tbl: catalog.TableVersion, stmt: sql.Select, refd_tbl_ids: Set[UUID] = {},
            exact_version_only: Set[UUID] = {}
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
        joined_tbls: List[catalog.TableVersion] = [tbl]
        while tbl.base is not None:
            tbl = tbl.base
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

    def _open(self) -> None:
        """Add PK columns to self.stmt"""
        if self.set_pk:
            assert self.ctx.pk_clause is not None
            # TODO: don't add pk columns if we're already retrieving them via RowidRefs
            pk_cols = self.ctx.pk_clause
            self.num_pk_cols = len(pk_cols)
            self.stmt = self.stmt.add_columns(*pk_cols)

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

        output_batch = DataRowBatch(self.tbl, self.row_builder)
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


class ExprEvalNode(ExecNode):
    """Materializes expressions
    """
    @dataclass
    class Cohort:
        """List of exprs that form an evaluation context and contain calls to at most one NOS function"""
        exprs: List[exprs.Expr]
        model_info: Optional[nos.common.ModelSpec]
        segment_ctxs: List[exprs.DataRowBuilder.EvalCtx]
        target_slot_idxs: List[int]

        # for NOS cohorts:
        nos_param_names: Optional[List[str]] = None
        scalar_nos_param_names: Optional[List[str]] = None

        # for models on images:

        img_param_pos: Optional[int] = None  # position of the image parameter in the function signature
        # for multi-resolution image models
        img_batch_params: List[nos.common.ObjectTypeInfo] = field(default_factory=list)
        # for single-resolution image models
        batch_size: int = 8
        img_size: Optional[Tuple[int, int]] = None  # W, H

        def __post_init__(self):
            if self.model_info is None:
                return
            nos_calls = [e for e in self.exprs if isinstance(e, exprs.FunctionCall) and e.is_nos_call()]
            assert len(nos_calls) <= 1
            nos_call = nos_calls[0] if len(nos_calls) > 0 else None
            self.nos_param_names = self.model_info.signature.get_inputs_spec().keys()
            self.scalar_nos_param_names = []

            # try to determine batch_size and img_size
            batch_size = sys.maxsize
            for pos, (param_name, type_info) in enumerate(self.model_info.signature.get_inputs_spec().items()):
                if isinstance(type_info, list):
                    assert isinstance(type_info[0].base_spec(), nos.common.ImageSpec)
                    # this is a multi-resolution image model
                    self.img_batch_params = type_info
                    self.img_param_pos = pos
                else:
                    if not type_info.is_batched():
                        self.scalar_nos_param_names.append(param_name)
                        if param_name not in nos_call.constant_args:
                            # this is a scalar parameter that is not constant, so we need to do batches of 1
                            batch_size = 1
                    else:
                        batch_size = min(batch_size, type_info.batch_size())

                    if isinstance(type_info.base_spec(), nos.common.ImageSpec):
                        # this is a single-resolution image model
                        if type_info.base_spec().shape is not None:
                            self.img_size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
                        self.img_param_pos = pos
                        self.img_batch_params = []

            if batch_size == sys.maxsize:
                # some reasonable default
                self.batch_size = 8
            else:
                self.batch_size = batch_size

        def is_multi_res_model(self) -> bool:
            return self.img_param_pos is not None and len(self.img_batch_params) > 0

        def get_batch_params(self, img_size: Tuple[int, int]) -> Tuple[int, Tuple[int, int]]:
            """Returns batch_size and img_size appropriate for the given image size"""
            if len(self.img_batch_params) > 0:
                input_res = img_size[0] * img_size[1]
                resolutions = [info.base_spec().shape[0] * info.base_spec().shape[1] for info in self.img_batch_params]
                deltas = [abs(res - input_res) for res in resolutions]
                idx = deltas.index(min(deltas))
                type_info = self.img_batch_params[idx]
                return type_info.batch_size(), (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
            else:
                return self.batch_size, self.img_size

    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: List[exprs.Expr], input_exprs: List[exprs.Expr],
            ignore_errors: bool, input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.input_exprs = input_exprs
        input_slot_idxs = {e.slot_idx for e in input_exprs}
        # we're only materializing exprs that are not already in the input
        self.target_exprs = [e for e in output_exprs if e.slot_idx not in input_slot_idxs]
        self.ignore_errors = ignore_errors  # if False, raise exc.ExprEvalError on error in _exec_cohort()
        self.pbar: Optional[tqdm] = None
        self.cohorts: List[List[ExprEvalNode.Cohort]] = []
        self._create_cohorts()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)
        # compute target exprs
        for cohort in self.cohorts:
            self._exec_cohort(cohort, input_batch)
        _logger.debug(f'ExprEvalNode: returning {len(input_batch)} rows')
        return input_batch

    def _open(self) -> None:
        if self.ctx.show_pbar:
            self.pbar = tqdm(total=len(self.target_exprs) * self.ctx.num_rows, desc='Computing cells', unit='cells')

    def _close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

    def _get_nos_info(self, expr: exprs.Expr) -> Optional[nos.common.ModelSpec]:
        """Get ModelSpec if expr is a call to a NOS function, else None."""
        if not isinstance(expr, exprs.FunctionCall):
            return None
        return FunctionRegistry.get().get_nos_info(expr.fn)

    def _is_nos_call(self, expr: exprs.Expr) -> bool:
        return self._get_nos_info(expr) is not None

    def _create_cohorts(self) -> None:
        all_exprs = self.row_builder.get_dependencies(self.target_exprs)
        # break up all_exprs into cohorts such that each cohort contains calls to at most one NOS function;
        # seed the cohorts with only the nos calls
        cohorts: List[List[exprs.Expr]] = []
        current_nos_function: Optional[Function] = None
        for e in all_exprs:
            if not self._is_nos_call(e):
                continue
            if current_nos_function is None or current_nos_function != e.fn:
                # create a new cohort
                cohorts.append([])
                current_nos_function = e.fn
            cohorts[-1].append(e)

        # expand the cohorts to include all exprs that are in the same evaluation context as the NOS calls;
        # cohorts are evaluated in order, so we can exclude the target slots from preceding cohorts and input slots
        exclude = set([e.slot_idx for e in self.input_exprs])
        all_target_slot_idxs = set([e.slot_idx for e in self.target_exprs])
        target_slot_idxs: List[List[int]] = []  # the ones materialized by each cohort
        for i in range(len(cohorts)):
            cohorts[i] = self.row_builder.get_dependencies(
                cohorts[i], exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude])
            target_slot_idxs.append(
                [e.slot_idx for e in cohorts[i] if e.slot_idx in all_target_slot_idxs])
            exclude.update(target_slot_idxs[-1])

        all_cohort_slot_idxs = set([e.slot_idx for cohort in cohorts for e in cohort])
        remaining_slot_idxs = set(all_target_slot_idxs) - all_cohort_slot_idxs
        if len(remaining_slot_idxs) > 0:
            cohorts.append(self.row_builder.get_dependencies(
                [self.row_builder.unique_exprs[slot_idx] for slot_idx in remaining_slot_idxs],
                exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude]))
            target_slot_idxs.append(list(remaining_slot_idxs))
        # we need to have captured all target slots at this point
        assert all_target_slot_idxs == set().union(*target_slot_idxs)

        for i in range(len(cohorts)):
            cohort = cohorts[i]
            # segment the cohort into sublists that contain either a single NOS function call or no NOS function calls
            # (i.e., only computed cols)
            assert len(cohort) > 0
            # create the first segment here, so we can avoid checking for an empty list in the loop
            segments = [[cohort[0]]]
            is_nos_segment = self._is_nos_call(cohort[0])
            model_info: Optional[nos.common.ModelSpec] = self._get_nos_info(cohort[0])
            for e in cohort[1:]:
                if self._is_nos_call(e):
                    segments.append([e])
                    is_nos_segment = True
                    model_info = self._get_nos_info(e)
                else:
                    if is_nos_segment:
                        # start a new segment
                        segments.append([])
                        is_nos_segment = False
                    segments[-1].append(e)

            # we create the EvalCtxs manually because create_eval_ctx() would repeat the dependencies of each segment
            segment_ctxs = [
                exprs.RowBuilder.EvalCtx(
                    slot_idxs=[e.slot_idx for e in s], exprs=s, target_slot_idxs=[], target_exprs=[])
                for s in segments
            ]
            cohort_info = self.Cohort(cohort, model_info, segment_ctxs, target_slot_idxs[i])
            self.cohorts.append(cohort_info)

    def _exec_cohort(self, cohort: Cohort, rows: DataRowBatch) -> None:
        """Compute the cohort for the entire input batch by dividing it up into sub-batches"""
        batch_start_idx = 0  # start row of the current sub-batch
        # for multi-resolution models, we re-assess the correct NOS batch size for each input batch
        verify_nos_batch_size = cohort.is_multi_res_model()
        while batch_start_idx < len(rows):
            num_batch_rows = min(cohort.batch_size, len(rows) - batch_start_idx)
            for segment_ctx in cohort.segment_ctxs:
                if not self._is_nos_call(segment_ctx.exprs[0]):
                    # compute batch row-wise
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        self.row_builder.eval(rows[row_idx], segment_ctx, self.ctx.profile, ignore_errors=self.ignore_errors)
                else:
                    fn_call = segment_ctx.exprs[0]
                    # make a batched NOS call
                    arg_batches = [[] for _ in range(len(fn_call.args))]
                    assert len(cohort.nos_param_names) == len(arg_batches)

                    valid_batch_idxs: List[int] = []  # rows with exceptions are not valid
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        row = rows[row_idx]
                        if row.has_exc(fn_call.slot_idx):
                            # one of our inputs had an exception, skip this row
                            continue
                        valid_batch_idxs.append(row_idx)
                        args, kwargs = fn_call._make_args(row)
                        assert len(kwargs) == 0
                        for i in range(len(args)):
                            arg_batches[i].append(args[i])
                    num_valid_batch_rows = len(valid_batch_idxs)

                    if verify_nos_batch_size:
                        # we need to choose a batch size based on the image size
                        sample_img = arg_batches[cohort.img_param_pos][0]
                        nos_batch_size, target_res = cohort.get_batch_params(sample_img.size)
                        verify_nos_batch_size = False
                    else:
                        nos_batch_size, target_res = cohort.batch_size, cohort.img_size

                    # if we need to rescale image args, and we're doing object detection, we need to rescale the
                    # bounding boxes as well
                    scale_factors = np.ndarray((num_valid_batch_rows, 2), dtype=np.float32)
                    if cohort.img_param_pos is not None:
                        # for now, NOS will only receive RGB images
                        arg_batches[cohort.img_param_pos] = \
                            [img.convert('RGB') for img in arg_batches[cohort.img_param_pos]]
                        if target_res is not None:
                            # we need to record the scale factors and resize the images;
                            # keep in mind that every image could have a different resolution
                            scale_factors[:, 0] = \
                                [img.size[0]/target_res[0] for img in arg_batches[cohort.img_param_pos]]
                            scale_factors[:, 1] = \
                                [img.size[1]/target_res[1] for img in arg_batches[cohort.img_param_pos]]
                            arg_batches[cohort.img_param_pos] = [
                                # only resize if necessary
                                img.resize(target_res) if img.size != target_res else img
                                for img in arg_batches[cohort.img_param_pos]
                            ]

                    num_remaining_batch_rows = num_valid_batch_rows
                    while num_remaining_batch_rows > 0:
                        # we make NOS calls in batches of nos_batch_size
                        num_nos_batch_rows = min(nos_batch_size, num_remaining_batch_rows)
                        nos_batch_offset = num_valid_batch_rows - num_remaining_batch_rows  # offset into args, not rows
                        kwargs = {
                            param_name: args[nos_batch_offset:nos_batch_offset + num_nos_batch_rows]
                            for param_name, args in zip(cohort.nos_param_names, arg_batches)
                        }
                        # fix up scalar parameters
                        kwargs.update(
                            {param_name: kwargs[param_name][0] for param_name in cohort.scalar_nos_param_names})
                        start_ts = time.perf_counter()
                        _logger.debug(
                            f'Running NOS task {cohort.model_info.task}: '
                            f'batch_size={num_nos_batch_rows} target_res={target_res}')
                        result = Env.get().nos_client.Run(
                            task=cohort.model_info.task, model_name=cohort.model_info.name, **kwargs)
                        self.ctx.profile.eval_time[fn_call.slot_idx] += time.perf_counter() - start_ts
                        self.ctx.profile.eval_count[fn_call.slot_idx] += num_nos_batch_rows

                        if cohort.model_info.task == nos.common.TaskType.OBJECT_DETECTION_2D and target_res is not None:
                            # we need to rescale the bounding boxes
                            result_bboxes = []  # workaround: result['bboxes'][*] is immutable
                            for i, bboxes in enumerate(result['bboxes']):
                                bboxes = np.copy(bboxes)
                                nos_batch_row_idx = nos_batch_offset + i
                                bboxes[:, 0] *= scale_factors[nos_batch_row_idx, 0]
                                bboxes[:, 1] *= scale_factors[nos_batch_row_idx, 1]
                                bboxes[:, 2] *= scale_factors[nos_batch_row_idx, 0]
                                bboxes[:, 3] *= scale_factors[nos_batch_row_idx, 1]
                                result_bboxes.append(bboxes)
                            result['bboxes'] = result_bboxes

                        if len(result) == 1:
                            key = list(result.keys())[0]
                            row_results = result[key]
                        else:
                            # we rearrange result into one dict per row
                            row_results = [
                                {k: v[i].tolist() for k, v in result.items()} for i in range(num_nos_batch_rows)
                            ]

                        # move the result into the row batch
                        for result_idx in range(len(row_results)):
                            row_idx = valid_batch_idxs[nos_batch_offset + result_idx]
                            row = rows[row_idx]
                            row[fn_call.slot_idx] = row_results[result_idx]

                        num_remaining_batch_rows -= num_nos_batch_rows

                    # switch to the NOS-recommended batch size
                    cohort.batch_size = nos_batch_size
                    cohort.img_size = target_res

            # make sure images for stored cols have been saved to files before moving on to the next batch
            rows.flush_imgs(
                slice(batch_start_idx, batch_start_idx + num_batch_rows), self.stored_img_cols, self.flushed_img_slots)
            if self.pbar is not None:
                self.pbar.update(num_batch_rows * len(cohort.target_slot_idxs))
            batch_start_idx += num_batch_rows


class InsertDataNode(ExecNode):
    """Outputs in-memory data as a row batch of a particular table"""
    def __init__(
            self, tbl: catalog.TableVersion, rows: List[List[Any]], row_column_pos: Dict[str, int],
            row_builder: exprs.RowBuilder, input_cols: List[exprs.ColumnSlotIdx], start_row_id: int,
    ):
        super().__init__(row_builder, [], [], None)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.input_rows = rows
        self.row_column_pos = row_column_pos  # col name -> idx of col in self.input_rows
        self.input_cols = input_cols
        self.start_row_id = start_row_id
        self.has_returned_data = False
        self.output_rows: Optional[DataRowBatch] = None

        # TODO: remove this with component views
        self.boto_client: Optional[Any] = None

    def _open(self) -> None:
        """Create row batch and populate with self.data"""

        for info in self.input_cols:
            assert info.col.name in self.row_column_pos

        # before anything, convert any literal images within the input rows into references
        # copy the input rows to avoid indirectly modifying the argument
        _input_rows = [row.copy() for row in self.input_rows]
        for info in self.input_cols:
            if info.col.col_type.is_image_type():
                col_idx = self.row_column_pos[info.col.name]
                for row_idx, input_row in enumerate(_input_rows):
                    val = input_row[col_idx]
                    if isinstance(val, bytes):
                        # we will save literal to a file here and use this path as the new value
                        valpath = str(ImageStore.get_path(self.tbl.id, info.col.id, self.tbl.version))
                        open(valpath, 'wb').write(val)
                        input_row[col_idx] = valpath

        self.input_rows = _input_rows

        self.output_rows = DataRowBatch(self.tbl, self.row_builder, len(self.input_rows))
        for info in self.input_cols:
            col_idx = self.row_column_pos[info.col.name]
            for row_idx, input_row in enumerate(self.input_rows):
                self.output_rows[row_idx][info.slot_idx] = input_row[col_idx]

        self.output_rows.set_row_ids([self.start_row_id + i for i in range(len(self.output_rows))])
        self.ctx.num_rows = len(self.output_rows)

    def _get_local_path(self, url: str) -> str:
        """Returns local path for url"""
        if url is None:
            return None
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme == '' or parsed.scheme == 'file':
            # local file path
            return parsed.path
        if parsed.scheme == 's3':
            from pixeltable.utils.s3 import get_client
            if self.boto_client is None:
                self.boto_client = get_client()
            tmp_path = str(Env.get().tmp_dir / os.path.basename(parsed.path))
            self.boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), tmp_path)
            return tmp_path
        assert False, f'Unsupported URL scheme: {parsed.scheme}'

    def __next__(self) -> DataRowBatch:
        if self.has_returned_data:
            raise StopIteration
        self.has_returned_data = True
        _logger.debug(f'InsertDataNode: created row batch with {len(self.output_rows)} output_rows')
        return self.output_rows


class CachePrefetchNode(ExecNode):
    """Brings files with external URLs into the cache

    TODO:
    - maintain a queue of row batches, in order to overlap download and evaluation
    - adapting the number of download threads at runtime to maximize throughput
    """
    def __init__(self, tbl_id: UUID, file_col_info: List[exprs.ColumnSlotIdx], input: ExecNode):
        # []: we don't have anything to evaluate
        super().__init__(input.row_builder, [], [], input)
        self.tbl_id = tbl_id
        self.file_col_info = file_col_info

        # clients for specific services are constructed as needed, because it's time-consuming
        self.boto_client: Optional[Any] = None
        self.boto_client_lock = threading.Lock()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)

        # collect external URLs that aren't already cached, and set DataRow.file_paths for those that are
        file_cache = FileCache.get()
        cache_misses: List[Tuple[exprs.DataRow, ColumnInfo]] = []
        missing_url_rows: Dict[str, List[exprs.DataRow]] = defaultdict(list)
        for row in input_batch:
            for info in self.file_col_info:
                url = row.file_urls[info.slot_idx]
                if url is None or row.file_paths[info.slot_idx] is not None:
                    # nothing to do
                    continue
                if url in missing_url_rows:
                    missing_url_rows[url].append(row)
                    continue
                local_path = file_cache.lookup(url)
                if local_path is None:
                    cache_misses.append((row, info))
                    missing_url_rows[url].append(row)
                else:
                    row.set_file_path(info.slot_idx, local_path)

        # download the cache misses in parallel
        # TODO: set max_workers to maximize throughput
        futures: Dict[concurrent.futures.Future, Tuple[exprs.DataRow, exprs.ColumnSlotIdx]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for row, info in cache_misses:
                futures[executor.submit(self._fetch_url, row.file_urls[info.slot_idx])] = (row, info)
            for future in concurrent.futures.as_completed(futures):
                # TODO:  does this need to deal with recoverable errors (such as retry after throttling)?
                tmp_path = future.result()
                row, info = futures[future]
                url = row.file_urls[info.slot_idx]
                local_path = file_cache.add(self.tbl_id, info.col.id, url, tmp_path)
                _logger.debug(f'PrefetchNode: cached {url} as {local_path}')
                for row in missing_url_rows[url]:
                    row.set_file_path(info.slot_idx, str(local_path))

        return input_batch

    def _fetch_url(self, url: str) -> str:
        """Fetches a remote URL into Env.tmp_dir and returns its path"""
        parsed = urllib.parse.urlparse(url)
        assert parsed.scheme != '' and parsed.scheme != 'file'
        if parsed.scheme == 's3':
            from pixeltable.utils.s3 import get_client
            with self.boto_client_lock:
                if self.boto_client is None:
                    self.boto_client = get_client()
            tmp_path = Env.get().tmp_dir / os.path.basename(parsed.path)
            self.boto_client.download_file(parsed.netloc, parsed.path.lstrip('/'), str(tmp_path))
            return tmp_path
        assert False, f'Unsupported URL scheme: {parsed.scheme}'


class ComponentIterationNode(ExecNode):
    """Expands each row from a base table into one row per component returned by an iterator

    Returns row batches of OUTPUT_BATCH_SIZE size.
    """
    OUTPUT_BATCH_SIZE = 1024

    def __init__(self, view: catalog.TableVersion, input: ExecNode):
        assert view.is_component_view()
        super().__init__(input.row_builder, [], [], input)
        self.view = view
        iterator_args = [view.iterator_args.copy()]
        self.row_builder.substitute_exprs(iterator_args)
        self.iterator_args = iterator_args[0]
        self.iterator_args_ctx = self.row_builder.create_eval_ctx([self.iterator_args])
        self.iterator_output_schema, self.unstored_column_names = self.view.iterator_cls.output_schema()
        self.iterator_output_fields = list(self.iterator_output_schema.keys())
        self.iterator_output_cols = \
            {field_name: self.view.cols_by_name[field_name] for field_name in self.iterator_output_fields}
        # referenced iterator output fields
        self.refd_output_slot_idxs = {
            e.col.name: e.slot_idx for e in self.row_builder.unique_exprs
            if isinstance(e, exprs.ColumnRef) and e.col.name in self.iterator_output_fields
        }
        self._output: Optional[Generator[DataRowBatch, None, None]] = None

    def _output_batches(self) -> Generator[DataRowBatch, None, None]:
        output_batch = DataRowBatch(self.view, self.row_builder)
        for input_batch in self.input:
            for input_row in input_batch:
                self.row_builder.eval(input_row, self.iterator_args_ctx)
                iterator_args = input_row[self.iterator_args.slot_idx]
                iterator = self.view.iterator_cls(**iterator_args)
                for pos, component_dict in enumerate(iterator):
                    output_row = output_batch.add_row()
                    input_row.copy(output_row)
                    # we're expanding the input and need to add the iterator position to the pk
                    pk = output_row.pk[:-1] + (pos,) + output_row.pk[-1:]
                    output_row.set_pk(pk)

                    # verify and copy component_dict fields to their respective slots in output_row
                    for field_name, field_val in component_dict.items():
                        if field_name not in self.iterator_output_fields:
                            raise exc.Error(
                                f'Invalid field name {field_name} in output of {self.view.iterator_cls.__name__}')
                        if field_name not in self.refd_output_slot_idxs:
                            # we can ignore this
                            continue
                        output_col = self.iterator_output_cols[field_name]
                        output_col.col_type.validate_literal(field_val)
                        output_row[self.refd_output_slot_idxs[field_name]] = field_val
                    if len(component_dict) != len(self.iterator_output_fields):
                        missing_fields = set(self.refd_output_slot_idxs.keys()) - set(component_dict.keys())
                        raise exc.Error(
                            f'Invalid output of {self.view.iterator_cls.__name__}: '
                            f'missing fields {", ".join(missing_fields)}')

                    if len(output_batch) == self.OUTPUT_BATCH_SIZE:
                        yield output_batch
                        output_batch = DataRowBatch(self.view, self.row_builder)

        if len(output_batch) > 0:
            yield output_batch

    def __next__(self) -> DataRowBatch:
        if self._output is None:
            self._output = self._output_batches()
        return next(self._output)

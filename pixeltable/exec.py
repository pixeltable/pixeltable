from __future__ import annotations
from typing import List, Iterator, Set, Dict, Any, Optional, Tuple, Iterable
from dataclasses import dataclass, field
import logging
import time
import abc
import io
import sys

import numpy as np
from tqdm.autonotebook import tqdm
import nos
import sqlalchemy as sql

from pixeltable import exprs
from pixeltable import catalog
from pixeltable.utils.imgstore import ImageStore
from pixeltable.function import Function, FunctionRegistry
from pixeltable.env import Env
from pixeltable.utils.video import FrameIterator
from pixeltable import exceptions as exc


_logger = logging.getLogger('pixeltable')

@dataclass
class ColumnInfo:
    """info for how to locate materialized column in DataRow"""
    col: catalog.Column
    slot_idx: int


class DataRowBatch:
    """
    Set of DataRows, indexed by rowid. Keeps track of image memory consumption and can flush images.
    """
    def __init__(self, table: catalog.Table, evaluator: exprs.Evaluator, len: int = 0):
        self.table_id = table.id
        self.table_version = table.version
        self.evaluator = evaluator
        self.rows = [evaluator.prepare() for _ in range(len)]
        self.img_slot_idxs = [e.slot_idx for e in evaluator.unique_exprs if e.col_type.is_image_type()]
        self.array_slot_idxs = [e.slot_idx for e in evaluator.unique_exprs if e.col_type.is_array_type()]

    def add_row(self, row: Optional[exprs.DataRow] = None) -> exprs.DataRow:
        row = self.evaluator.prepare() if row is None else row
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        return self.rows.pop()

    def set_row_ids(self, row_ids: List[int]) -> None:
        assert len(row_ids) == len(self.rows)
        for row, row_id in zip(self.rows, row_ids):
            row.set_pk(row_id, self.table_version)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: object) -> exprs.DataRow:
        return self.rows[index]

    def __setitem__(self, index: object, val: Any) -> None:
        row_idx, slot_idx = index
        if slot_idx in self.img_slot_idxs and isinstance(val, str):
            # this is a filepath to an image
            row = self.rows[row_idx]
            assert row.img_files[slot_idx] is None
            row.img_files[slot_idx] = val
            row.has_val[slot_idx] = True
        elif slot_idx in self.array_slot_idxs and isinstance(val, bytes):
            self.rows[row_idx][slot_idx] = np.load(io.BytesIO(val))
        else:
            self.rows[row_idx][slot_idx] = val

    def flush_imgs(
            self, idx_range: Optional[slice] = None, stored_img_info: List[ColumnInfo] = [],
            flushed_slot_idxs: List[int] = []
    ) -> None:
        """Flushes images in the given range of rows."""
        if len(stored_img_info) == 0 and len(flushed_slot_idxs) == 0:
            return
        if idx_range is None:
            idx_range = slice(0, len(self.rows))
        for row in self.rows[idx_range]:
            for info in stored_img_info:
                filepath = str(ImageStore.get_path(self.table_id, info.col.id, row.v_min, row.row_id, 'jpg'))
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
    def __init__(self, evaluator: exprs.Evaluator, *, show_pbar: bool = False, batch_size: int = 0):
        self.show_pbar = show_pbar
        self.batch_size = batch_size
        self.profile = exprs.ExecProfile(evaluator)
        # num_rows is used to compute the total number of computed cells used for the progress bar
        self.num_rows: Optional[int] = None


class ExecNode(abc.ABC):
    """Base class of all execution nodes"""
    def __init__(
            self, evaluator: exprs.Evaluator, output_exprs: Iterable[exprs.Expr], input_exprs: Iterable[exprs.Expr],
            input: Optional[ExecNode] = None):
        self.evaluator = evaluator
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        dependencies = evaluator.get_eval_ctx(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in dependencies
            if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.stored_img_cols: List[ColumnInfo] = []
        self.ctx: Optional[ExecContext] = None  # all nodes of a tree share the same context

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def set_stored_img_cols(self, stored_img_cols: List[ColumnInfo]) -> None:
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
            self, tbl: catalog.Table, evaluator: exprs.Evaluator, group_by: List[exprs.Expr],
            agg_fn_calls: List[exprs.FunctionCall], input_exprs: List[exprs.Expr], input: ExecNode
    ):
        super().__init__(evaluator, group_by + agg_fn_calls, input_exprs, input)
        self.input = input
        self.evaluator = evaluator
        self.group_by = group_by
        self.input_exprs = input_exprs
        self.agg_fn_calls = agg_fn_calls
        self.agg_fn_eval_ctx = evaluator.get_eval_ctx(agg_fn_calls, exclude=input_exprs)
        self.output_batch = DataRowBatch(tbl, evaluator, 0)

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
                    self.evaluator.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
                    self.output_batch.add_row(prev_row)
                    current_group = group
                    self._reset_agg_state(0)
                self._update_agg_state(row, 0)
                prev_row = row
        # emit the last group
        self.evaluator.eval(prev_row, self.agg_fn_eval_ctx, profile=self.ctx.profile)
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
            self, tbl: catalog.Table, evaluator: exprs.Evaluator, sql_exprs: Iterable[exprs.Expr],
            where_clause: Optional[sql.sql.expression.ClauseElement] = None, filter: Optional[exprs.Predicate] = None,
            order_by_clause: List[sql.sql.expression.ClauseElement] = [],
            limit: int = 0, set_pk: bool = False, rowids: List[int] = []

    ):
        """
        Args:
            sql_exprs: list of exprs for which sql_expr() is not None
            sql_where_clause: SQL where clause
            filter: additional Where-clause predicate that can't be evaluated via SQL
            limit: max number of rows to return: 0 = no limit
            set_pk: if True, sets the primary for each DataRow
            rowids: if not empty, only return rows with these rowids, in that order
        """
        # create Select stmt
        super().__init__(evaluator, sql_exprs, [], None)
        self.tbl = tbl
        self.evaluator = evaluator
        self.sql_exprs = sql_exprs
        self.set_pk = set_pk or len(rowids) > 0  # we also need to set the rowid if we're ordering by rowid
        self.filter = filter
        self.filter_eval_ctx = evaluator.get_eval_ctx([filter], exclude=sql_exprs) if filter is not None else []
        self.limit = limit
        self.rowids = rowids
        select_list = [e.sql_expr() for e in sql_exprs]
        if self.set_pk:
            select_list.extend([tbl.rowid_col, tbl.v_min_col])
        self.stmt = sql.select(*select_list) \
            .where(tbl.v_min_col <= tbl.version) \
            .where(tbl.v_max_col > tbl.version)
        if where_clause is not None:
            self.stmt = self.stmt.where(where_clause)
        if len(rowids) > 0:
            self.stmt = self.stmt.where(self.tbl.rowid_col.in_(rowids))
        if len(order_by_clause) > 0:
            self.stmt = self.stmt.order_by(*order_by_clause)
        if limit != 0 and self.filter is None:
            # if we need to do post-SQL filtering, we can't use LIMIT
            self.stmt = self.stmt.limit(limit)

        self.conn: Optional[sql.engine.Connection] = None
        self.result_cursor: Optional[sql.engine.CursorResult] = None

    def __next__(self) -> DataRowBatch:
        if self.conn is None:
            self.conn = Env.get().engine.connect()
            try:
                # run the query; do this here rather than in _open(), exceptions are only expected during iteration
                self.result_cursor = self.conn.execute(self.stmt)
                self.has_more_rows = True
            except Exception as e:
                self.conn.close()
                self.conn = None
                self.has_more_rows = False
                raise e

        if not self.has_more_rows:
            raise StopIteration

        output_batch = DataRowBatch(self.tbl, self.evaluator)
        needs_row = True
        while self.ctx.batch_size == 0 or len(output_batch) < self.ctx.batch_size:
            try:
                sql_row = next(self.result_cursor)
            except StopIteration:
                self.has_more_rows = False
                break

            if needs_row:
                output_batch.add_row()
            if self.set_pk:
                row = output_batch[-1]
                row.set_pk(sql_row[-2], sql_row[-1])
            # copy the output of the SQL query into the output row
            for i, e in enumerate(self.sql_exprs):
                slot_idx = e.slot_idx
                output_batch[-1, slot_idx] = sql_row[i]
            if self.filter is not None:
                row = output_batch[-1]
                self.evaluator.eval(row, self.filter_eval_ctx, profile=self.ctx.profile)
                if row[self.filter.slot_idx]:
                    needs_row = True
                    if self.limit is not None and len(output_batch) >= self.limit:
                        self.has_more_rows = False
                        break
                else:
                    # we re-use this row for the next sql row if it didn't pass the filter
                    needs_row = False
                    row.clear()

        if not needs_row:
            # the last row didn't pass the filter
            assert self.filter is not None
            output_batch.pop_row()

        output_batch.flush_imgs(None, self.stored_img_cols, self.flushed_img_slots)
        if self.rowids:
            # we need to reorder the output batch to match the order in self.rowids
            rowid_to_idx = {row.row_id: i for i, row in enumerate(output_batch)}
            reordered_batch = DataRowBatch(self.tbl, self.evaluator)
            for rowid in self.rowids:
                if rowid not in rowid_to_idx:
                    continue
                reordered_batch.add_row(output_batch[rowid_to_idx[rowid]])
            output_batch = reordered_batch

        _logger.debug(f'SqlScanNode: returning {len(output_batch)} rows')
        return output_batch

    def _close(self) -> None:
        if self.result_cursor is not None:
            self.result_cursor.close()
        if self.conn is not None:
            self.conn.close()


class ExprEvalNode(ExecNode):
    """Materializes expressions
    """
    @dataclass
    class Cohort:
        """List of exprs that form an evaluation context and contain calls to at most one NOS function"""
        exprs: List[exprs.Expr]
        model_info: Optional[nos.common.ModelSpec]
        segments: List[List[exprs.Expr]]
        target_slot_idxs: List[int]
        nos_param_names: Optional[List[str]] = None

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
            self.nos_param_names = self.model_info.signature.get_inputs_spec().keys()
            # try to determine batch_size and img_size
            for pos, (_, type_info) in enumerate(self.model_info.signature.get_inputs_spec().items()):
                if isinstance(type_info, list):
                    if isinstance(type_info[0].base_spec(), nos.common.ImageSpec):
                        # this is a multi-resolution image model
                        self.img_batch_params = type_info
                        self.img_param_pos = pos
                        return
                    else:
                        continue
                if isinstance(type_info.base_spec(), nos.common.ImageSpec):
                    # this is a single-resolution image model
                    if type_info.base_spec().shape is not None:
                        self.img_size = (type_info.base_spec().shape[1], type_info.base_spec().shape[0])
                    if type_info.batch_size() is not None:
                        self.batch_size = type_info.batch_size()
                    self.img_param_pos = pos
                    self.img_batch_params = []
                    return

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
            self, evaluator: exprs.Evaluator, output_exprs: List[exprs.Expr], input_exprs: List[exprs.Expr],
            ignore_errors: bool, input: ExecNode
    ):
        super().__init__(evaluator, output_exprs, input_exprs, input)
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
        if len(input_batch) == 0:
            return input_batch
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
        all_exprs = self.evaluator.get_eval_ctx(self.target_exprs)
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
            cohorts[i] = self.evaluator.get_eval_ctx(
                cohorts[i], exclude = [self.evaluator.unique_exprs[slot_idx] for slot_idx in exclude])
            target_slot_idxs.append(
                [e.slot_idx for e in cohorts[i] if e.slot_idx in all_target_slot_idxs])
            exclude.update(target_slot_idxs[-1])

        all_cohort_slot_idxs = set([e.slot_idx for cohort in cohorts for e in cohort])
        remaining_slot_idxs = set(all_target_slot_idxs) - all_cohort_slot_idxs
        if len(remaining_slot_idxs) > 0:
            cohorts.append(self.evaluator.get_eval_ctx(
                [self.evaluator.unique_exprs[slot_idx] for slot_idx in remaining_slot_idxs],
                exclude=[self.evaluator.unique_exprs[slot_idx] for slot_idx in exclude]))
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
            cohort_info = self.Cohort(cohort, model_info, segments, target_slot_idxs[i])
            self.cohorts.append(cohort_info)

    def _exec_cohort(self, cohort: Cohort, rows: DataRowBatch) -> None:
        """Compute the cohort for the entire input batch by dividing it up into sub-batches"""
        batch_start_idx = 0  # start row of the current sub-batch
        # for multi-resolution models, we re-assess the correct NOS batch size for each input batch
        verify_nos_batch_size = cohort.is_multi_res_model()
        while batch_start_idx < len(rows):
            num_batch_rows = min(cohort.batch_size, len(rows) - batch_start_idx)
            for segment in cohort.segments:
                if not self._is_nos_call(segment[0]):
                    # compute batch row-wise
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        self.evaluator.eval(rows[row_idx], segment, self.ctx.profile, ignore_errors=self.ignore_errors)
                else:
                    fn_call = segment[0]
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
            self, tbl: catalog.MutableTable, rows: List[List[Any]], row_column_pos: Dict[str, int],
            evaluator: exprs.Evaluator, input_cols: List[ColumnInfo], frame_idx_slot_idx: int, start_row_id: int,
    ):
        super().__init__(evaluator, [], [], None)
        self.tbl = tbl
        self.input_rows = rows
        self.row_column_pos = row_column_pos  # col name -> idx of col in self.input_rows
        self.evaluator = evaluator
        self.input_cols = input_cols
        self.frame_idx_slot_idx = frame_idx_slot_idx
        self.start_row_id = start_row_id
        self.has_returned_data = False
        self.output_rows: Optional[DataRowBatch] = None

    def _open(self) -> None:
        """Create row batch and populate with self.data"""
        if not self.tbl.extracts_frames():
            self.output_rows = DataRowBatch(self.tbl, self.evaluator, len(self.input_rows))
            for info in self.input_cols:
                assert info.col.name in self.row_column_pos
                col_idx = self.row_column_pos[info.col.name]
                for row_idx, input_row in enumerate(self.input_rows):
                    self.output_rows[row_idx, info.slot_idx] = input_row[col_idx]
        else:
            # we're extracting frames: we replace each row with one row per frame, which has the frame_idx col set
            video_col = self.tbl.frame_src_col()
            assert video_col is not None and self.frame_idx_slot_idx is not None
            # get the output row count per video
            counts = [0] * len(self.input_rows)
            assert video_col.name in self.row_column_pos
            video_col_idx = self.row_column_pos[video_col.name]
            for row_idx, input_row in enumerate(self.input_rows):
                video_path = input_row[video_col_idx]
                if video_path is None:
                    counts[row_idx] = 1  # this will occupy one row in output_rows
                    continue
                with FrameIterator(video_path, fps=self.tbl.parameters.extraction_fps) as frame_iter:
                    counts[row_idx] = len(frame_iter)

            self.output_rows = DataRowBatch(self.tbl, self.evaluator, sum(counts))
            # populate frame_idx_col
            row_idx = 0
            for input_row_idx, count in enumerate(counts):
                if self.input_rows[video_col_idx] is None:
                    self.output_rows[row_idx, self.frame_idx_slot_idx] = None
                    row_idx += 1
                    continue
                for i in range(count):
                    self.output_rows[row_idx, self.frame_idx_slot_idx] = i
                    row_idx += 1
            # populate cols we get from the input df
            for info in self.input_cols:
                row_idx = 0
                col_idx = self.row_column_pos[info.col.name]
                for input_row_idx, input_row in enumerate(self.input_rows):
                    val = input_row[col_idx]
                    for i in range(counts[input_row_idx]):
                        self.output_rows[row_idx, info.slot_idx] = val
                        row_idx += 1

        # assign row ids
        self.output_rows.set_row_ids([self.start_row_id + i for i in range(len(self.output_rows))])
        self.ctx.num_rows = len(self.output_rows)

    def __next__(self) -> DataRowBatch:
        if self.has_returned_data:
            raise StopIteration
        self.has_returned_data = True
        _logger.debug(f'InsertDataNode: created row batch with {len(self.output_rows)} output_rows')
        return self.output_rows

import logging
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

from tqdm import TqdmWarning, tqdm

from pixeltable import exprs
from pixeltable.func import CallableFunction

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class ExprEvalNode(ExecNode):
    """Materializes expressions"""

    @dataclass
    class Cohort:
        """List of exprs that form an evaluation context and contain calls to at most one external function"""

        exprs_: list[exprs.Expr]
        batched_fn: Optional[CallableFunction]
        segment_ctxs: list['exprs.RowBuilder.EvalCtx']
        target_slot_idxs: list[int]
        batch_size: int = 8

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode,
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.input_exprs = input_exprs
        input_slot_idxs = {e.slot_idx for e in input_exprs}
        # we're only materializing exprs that are not already in the input
        self.target_exprs = [e for e in output_exprs if e.slot_idx not in input_slot_idxs]
        self.pbar: Optional[tqdm] = None
        self.cohorts: list[ExprEvalNode.Cohort] = []
        self._create_cohorts()

    def __next__(self) -> DataRowBatch:
        input_batch = next(self.input)
        # compute target exprs
        for cohort in self.cohorts:
            self._exec_cohort(cohort, input_batch)
        _logger.debug(f'ExprEvalNode: returning {len(input_batch)} rows')
        return input_batch

    def _open(self) -> None:
        warnings.simplefilter('ignore', category=TqdmWarning)
        # This is a temporary hack. When B-tree indices on string columns were implemented (via computed columns
        # that invoke the `BtreeIndex.str_filter` udf), it resulted in frivolous progress bars appearing on every
        # insertion. This special-cases the `str_filter` call to suppress the corresponding progress bar.
        # TODO(aaron-siegel) Remove this hack once we clean up progress bars more generally.
        is_str_filter_node = all(
            isinstance(expr, exprs.FunctionCall) and expr.fn.name == 'str_filter' for expr in self.output_exprs
        )
        if self.ctx.show_pbar and not is_str_filter_node:
            self.pbar = tqdm(
                total=len(self.target_exprs) * self.ctx.num_rows,
                desc='Computing cells',
                unit=' cells',
                ncols=100,
                file=sys.stdout,
            )

    def _close(self) -> None:
        if self.pbar is not None:
            self.pbar.close()

    def _get_batched_fn(self, expr: exprs.Expr) -> Optional[CallableFunction]:
        if isinstance(expr, exprs.FunctionCall) and isinstance(expr.fn, CallableFunction) and expr.fn.is_batched:
            return expr.fn
        return None

    def _is_batched_fn_call(self, expr: exprs.Expr) -> bool:
        return self._get_batched_fn(expr) is not None

    def _create_cohorts(self) -> None:
        all_exprs = self.row_builder.get_dependencies(self.target_exprs)
        # break up all_exprs into cohorts such that each cohort contains calls to at most one external function;
        # seed the cohorts with only the ext fn calls
        cohorts: list[list[exprs.Expr]] = []
        current_batched_fn: Optional[CallableFunction] = None
        for e in all_exprs:
            if not self._is_batched_fn_call(e):
                continue
            assert isinstance(e, exprs.FunctionCall)
            assert isinstance(e.fn, CallableFunction)
            if current_batched_fn is None or current_batched_fn != e.fn:
                # create a new cohort
                cohorts.append([])
                current_batched_fn = e.fn
            cohorts[-1].append(e)

        # expand the cohorts to include all exprs that are in the same evaluation context as the external calls;
        # cohorts are evaluated in order, so we can exclude the target slots from preceding cohorts and input slots
        exclude = set(e.slot_idx for e in self.input_exprs)
        all_target_slot_idxs = set(e.slot_idx for e in self.target_exprs)
        target_slot_idxs: list[list[int]] = []  # the ones materialized by each cohort
        for i in range(len(cohorts)):
            cohorts[i] = self.row_builder.get_dependencies(
                cohorts[i], exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude]
            )
            target_slot_idxs.append([e.slot_idx for e in cohorts[i] if e.slot_idx in all_target_slot_idxs])
            exclude.update(target_slot_idxs[-1])

        all_cohort_slot_idxs = set(e.slot_idx for cohort in cohorts for e in cohort)
        remaining_slot_idxs = set(all_target_slot_idxs) - all_cohort_slot_idxs
        if len(remaining_slot_idxs) > 0:
            cohorts.append(
                self.row_builder.get_dependencies(
                    [self.row_builder.unique_exprs[slot_idx] for slot_idx in remaining_slot_idxs],
                    exclude=[self.row_builder.unique_exprs[slot_idx] for slot_idx in exclude],
                )
            )
            target_slot_idxs.append(list(remaining_slot_idxs))
        # we need to have captured all target slots at this point
        assert all_target_slot_idxs == set().union(*target_slot_idxs)

        for i in range(len(cohorts)):
            cohort = cohorts[i]
            # segment the cohort into sublists that contain either a single ext. function call or no ext. function calls
            # (i.e., only computed cols)
            assert len(cohort) > 0
            # create the first segment here, so we can avoid checking for an empty list in the loop
            segments = [[cohort[0]]]
            is_batched_segment = self._is_batched_fn_call(cohort[0])
            batched_fn: Optional[CallableFunction] = self._get_batched_fn(cohort[0])
            for e in cohort[1:]:
                if self._is_batched_fn_call(e):
                    segments.append([e])
                    is_batched_segment = True
                    batched_fn = self._get_batched_fn(e)
                else:
                    if is_batched_segment:
                        # start a new segment
                        segments.append([])
                        is_batched_segment = False
                    segments[-1].append(e)

            # we create the EvalCtxs manually because create_eval_ctx() would repeat the dependencies of each segment
            segment_ctxs = [
                exprs.RowBuilder.EvalCtx(
                    slot_idxs=[e.slot_idx for e in s], exprs=s, target_slot_idxs=[], target_exprs=[]
                )
                for s in segments
            ]
            cohort_info = self.Cohort(cohort, batched_fn, segment_ctxs, target_slot_idxs[i])
            self.cohorts.append(cohort_info)

    def _exec_cohort(self, cohort: Cohort, rows: DataRowBatch) -> None:
        """Compute the cohort for the entire input batch by dividing it up into sub-batches"""
        batch_start_idx = 0  # start row of the current sub-batch
        # for multi-resolution models, we re-assess the correct ext fn batch size for each input batch
        ext_batch_size = cohort.batched_fn.get_batch_size() if cohort.batched_fn is not None else None
        if ext_batch_size is not None:
            cohort.batch_size = ext_batch_size

        while batch_start_idx < len(rows):
            num_batch_rows = min(cohort.batch_size, len(rows) - batch_start_idx)
            for segment_ctx in cohort.segment_ctxs:
                if not self._is_batched_fn_call(segment_ctx.exprs[0]):
                    # compute batch row-wise
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        self.row_builder.eval(
                            rows[row_idx], segment_ctx, self.ctx.profile, ignore_errors=self.ctx.ignore_errors
                        )
                else:
                    fn_call = segment_ctx.exprs[0]
                    assert isinstance(fn_call, exprs.FunctionCall)
                    # make a batched external function call
                    arg_batches: list[list[exprs.Expr]] = [[] for _ in range(len(fn_call.args))]
                    kwarg_batches: dict[str, list[exprs.Expr]] = {k: [] for k in fn_call.kwargs.keys()}

                    valid_batch_idxs: list[int] = []  # rows with exceptions are not valid
                    for row_idx in range(batch_start_idx, batch_start_idx + num_batch_rows):
                        row = rows[row_idx]
                        if row.has_exc(fn_call.slot_idx):
                            # one of our inputs had an exception, skip this row
                            continue
                        valid_batch_idxs.append(row_idx)
                        args, kwargs = fn_call._make_args(row)
                        for i in range(len(args)):
                            arg_batches[i].append(args[i])
                        for k in kwargs.keys():
                            kwarg_batches[k].append(kwargs[k])
                    num_valid_batch_rows = len(valid_batch_idxs)

                    if ext_batch_size is None:
                        # we need to choose a batch size based on the args
                        assert isinstance(fn_call.fn, CallableFunction)
                        sample_args = [arg_batches[i][0] for i in range(len(arg_batches))]
                        ext_batch_size = fn_call.fn.get_batch_size(*sample_args)

                    num_remaining_batch_rows = num_valid_batch_rows
                    while num_remaining_batch_rows > 0:
                        # we make ext. fn calls in batches of ext_batch_size
                        if ext_batch_size is None:
                            pass
                        num_ext_batch_rows = min(ext_batch_size, num_remaining_batch_rows)
                        ext_batch_offset = num_valid_batch_rows - num_remaining_batch_rows  # offset into args, not rows
                        call_args = [
                            arg_batches[i][ext_batch_offset : ext_batch_offset + num_ext_batch_rows]
                            for i in range(len(arg_batches))
                        ]
                        call_kwargs = {
                            k: kwarg_batches[k][ext_batch_offset : ext_batch_offset + num_ext_batch_rows]
                            for k in kwarg_batches.keys()
                        }
                        start_ts = time.perf_counter()
                        assert isinstance(fn_call.fn, CallableFunction)
                        result_batch = fn_call.fn.exec_batch(call_args, call_kwargs)
                        self.ctx.profile.eval_time[fn_call.slot_idx] += time.perf_counter() - start_ts
                        self.ctx.profile.eval_count[fn_call.slot_idx] += num_ext_batch_rows

                        # move the result into the row batch
                        for result_idx in range(len(result_batch)):
                            row_idx = valid_batch_idxs[ext_batch_offset + result_idx]
                            row = rows[row_idx]
                            row[fn_call.slot_idx] = result_batch[result_idx]

                        num_remaining_batch_rows -= num_ext_batch_rows

                    # switch to the ext fn batch size
                    cohort.batch_size = ext_batch_size

            # make sure images for stored cols have been saved to files before moving on to the next batch
            rows.flush_imgs(
                slice(batch_start_idx, batch_start_idx + num_batch_rows), self.stored_img_cols, self.flushed_img_slots
            )
            if self.pbar is not None:
                self.pbar.update(num_batch_rows * len(cohort.target_slot_idxs))
            batch_start_idx += num_batch_rows

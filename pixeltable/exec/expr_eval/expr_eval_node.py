from __future__ import annotations

import asyncio
import logging
import traceback
from types import TracebackType
from typing import AsyncIterator, Iterable

import numpy as np

import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable.utils.progress_reporter import ProgressReporter

from ..data_row_batch import DataRowBatch
from ..exec_node import ExecNode
from .evaluators import FnCallEvaluator, NestedRowList
from .globals import ExprEvalCtx, Scheduler
from .row_buffer import RowBuffer
from .schedulers import SCHEDULERS

_logger = logging.getLogger('pixeltable')


class ExprEvalNode(ExecNode):
    """
    Expression evaluation

    Resource management:
    - the execution system tries to limit total memory consumption by limiting the number of rows that are in
      circulation
    - during execution, slots that aren't part of the output are garbage collected as soon as their direct dependents
      are materialized

    TODO:
    - Literal handling: currently, Literal values are copied into slots via the normal evaluation mechanism, which is
      needless overhead; instead: pre-populate Literal slots in _init_row()
    - dynamically determine MAX_BUFFERED_ROWS, based on the avg memory consumption of a row and our configured memory
      limit
    - local model inference on gpu: currently, no attempt is made to ensure that models can fit onto the gpu
      simultaneously, which will cause errors; instead, the execution should be divided into sequential phases, each
      of which only contains a subset of the models which is known to fit onto the gpu simultaneously
    """

    maintain_input_order: bool  # True if we're returning rows in the order we received them from our input
    outputs: np.ndarray  # bool per slot; True if this slot is part of our output
    schedulers: dict[str, Scheduler]  # key: resource pool name
    eval_ctx: ExprEvalCtx  # for input/output rows

    # execution state
    tasks: set[asyncio.Task]  # collects all running tasks to prevent them from getting gc'd
    exc_event: asyncio.Event  # set if an exception needs to be propagated
    error: Exception | None  # exception that needs to be propagated
    completed_rows: asyncio.Queue[exprs.DataRow]  # rows that have completed evaluation
    completed_event: asyncio.Event  # set when completed_rows is non-empty
    input_iter: AsyncIterator[DataRowBatch]
    current_input_batch: DataRowBatch | None  # batch from which we're currently consuming rows
    input_row_idx: int  # next row to consume from current_input_batch
    next_input_batch: DataRowBatch | None  # read-ahead input batch
    avail_input_rows: int  # total number across both current_/next_input_batch
    input_complete: bool  # True if we've received all input batches
    num_in_flight: int  # number of dispatched rows that haven't completed
    row_pos_map: dict[int, int] | None  # id(row) -> position of row in input; only set if maintain_input_order
    output_buffer: RowBuffer  # holds rows that are ready to be returned, in order
    progress_reporter: ProgressReporter | None

    # debugging
    num_input_rows: int
    num_output_rows: int

    BATCH_SIZE = 64
    MAX_BUFFERED_ROWS = 2048  # maximum number of rows that have been dispatched but not yet returned

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode,
        maintain_input_order: bool = True,
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.maintain_input_order = maintain_input_order
        self.outputs = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.outputs[output_slot_idxs] = True
        self.tasks = set()
        self.error = None

        self.input_iter = self.input.__aiter__()
        self.current_input_batch = None
        self.next_input_batch = None
        self.input_row_idx = 0
        self.avail_input_rows = 0
        self.input_complete = False
        self.num_in_flight = 0
        self.row_pos_map = None
        self.output_buffer = RowBuffer(self.MAX_BUFFERED_ROWS)
        self.progress_reporter = None

        self.num_input_rows = 0
        self.num_output_rows = 0

        # self.slot_evaluators = {}
        self.schedulers = {}
        # self._init_slot_evaluators()
        self.eval_ctx = ExprEvalCtx(self, self.row_builder, output_exprs, input_exprs)

    def _open(self) -> None:
        self.progress_reporter = self.ctx.add_progress_reporter('Cell computations', 'cells')

    def set_input_order(self, maintain_input_order: bool) -> None:
        self.maintain_input_order = maintain_input_order

    async def _fetch_input_batch(self) -> None:
        """
        Fetches another batch from our input or sets input_complete to True if there are no more batches.

        - stores the batch in current_input_batch, if not already set, or next_input_batch
        - updates row_pos_map, if needed
        """
        assert not self.input_complete
        try:
            batch = await anext(self.input_iter)
            if self.progress_reporter is not None:
                # make sure our progress reporter shows up before we run anything long
                self.progress_reporter.update(0)
            assert self.next_input_batch is None
            if self.current_input_batch is None:
                self.current_input_batch = batch
            else:
                self.next_input_batch = batch
            if self.maintain_input_order:
                for idx, row in enumerate(batch.rows):
                    self.row_pos_map[id(row)] = self.num_input_rows + idx
            self.num_input_rows += len(batch)
            self.avail_input_rows += len(batch)
            _logger.debug(
                f'adding input: batch_size={len(batch)} #input_rows={self.num_input_rows} '
                f'#avail={self.avail_input_rows}'
            )
        except StopAsyncIteration:
            self.input_complete = True
            _logger.debug(f'finished input: #input_rows={self.num_input_rows}, #avail={self.avail_input_rows}')
        # make sure to pass DBAPIError through, so the transaction handling logic sees it
        except Exception as exc:
            self.error = exc
            self.exc_event.set()

    @property
    def total_buffered(self) -> int:
        return self.num_in_flight + self.completed_rows.qsize() + self.output_buffer.num_rows

    def _dispatch_input_rows(self) -> None:
        """Dispatch the maximum number of input rows, given total_buffered; does not block"""
        if self.avail_input_rows == 0:
            return
        num_rows = min(self.MAX_BUFFERED_ROWS - self.total_buffered, self.avail_input_rows)
        assert num_rows >= 0
        if num_rows == 0:
            return
        assert self.current_input_batch is not None
        avail_current_batch_rows = len(self.current_input_batch) - self.input_row_idx

        rows: list[exprs.DataRow]
        if avail_current_batch_rows > num_rows:
            # we only need rows from current_input_batch
            rows = self.current_input_batch.rows[self.input_row_idx : self.input_row_idx + num_rows]
            self.input_row_idx += num_rows
        else:
            # we need rows from both current_/next_input_batch
            rows = self.current_input_batch.rows[self.input_row_idx :]
            self.current_input_batch = self.next_input_batch
            self.next_input_batch = None
            self.input_row_idx = 0
            num_remaining = num_rows - len(rows)
            if num_remaining > 0:
                rows.extend(self.current_input_batch.rows[:num_remaining])
                self.input_row_idx = num_remaining
        self.avail_input_rows -= num_rows
        self.num_in_flight += num_rows
        self._log_state(f'dispatch input ({num_rows})')

        self.eval_ctx.init_rows(rows)
        self.dispatch(rows, self.eval_ctx)

    def _log_state(self, prefix: str) -> None:
        _logger.debug(
            f'{prefix}: #in-flight={self.num_in_flight} #complete={self.completed_rows.qsize()} '
            f'#output-buffer={self.output_buffer.num_rows} #ready={self.output_buffer.num_ready} '
            f'total-buffered={self.total_buffered} #avail={self.avail_input_rows} '
            f'#input={self.num_input_rows} #output={self.num_output_rows}'
        )

    def _init_schedulers(self) -> None:
        resource_pools = {
            eval.fn_call.resource_pool
            for eval in self.eval_ctx.slot_evaluators.values()
            if isinstance(eval, FnCallEvaluator)
        }
        resource_pools = {pool for pool in resource_pools if pool is not None}
        for pool_name in resource_pools:
            for scheduler in SCHEDULERS:
                if scheduler.matches(pool_name):
                    self.schedulers[pool_name] = scheduler(pool_name, self)
                    break
            if pool_name not in self.schedulers:
                raise RuntimeError(f'No scheduler found for resource pool {pool_name}')

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        """
        Main event loop

        Goals:
        - return completed DataRowBatches as soon as they become available
        - maximize the number of rows in flight in order to maximize parallelism, up to the given limit
        """
        # initialize completed_rows and events, now that we have the correct event loop
        self.completed_rows = asyncio.Queue[exprs.DataRow]()
        self.exc_event = asyncio.Event()
        self.completed_event = asyncio.Event()
        self._init_schedulers()
        if self.maintain_input_order:
            self.row_pos_map = {}
            self.output_buffer.set_row_pos_map(self.row_pos_map)

        row: exprs.DataRow
        exc_event_aw = asyncio.create_task(self.exc_event.wait(), name='exc_event.wait()')
        input_batch_aw: asyncio.Task | None = None
        completed_aw: asyncio.Task | None = None
        closed_evaluators = False  # True after calling Evaluator.close()
        exprs.Expr.prepare_list(self.eval_ctx.all_exprs)

        try:
            while True:
                # process completed rows before doing anything else
                while not self.completed_rows.empty():
                    # move completed rows to output buffer
                    while not self.completed_rows.empty():
                        row = self.completed_rows.get_nowait()
                        self.output_buffer.add_row(row)
                        if self.row_pos_map is not None:
                            self.row_pos_map.pop(id(row))

                    self._log_state('processed completed')
                    # return as many batches as we have available
                    while self.output_buffer.num_ready >= self.BATCH_SIZE:
                        batch_rows = self.output_buffer.get_rows(self.BATCH_SIZE)
                        self.num_output_rows += len(batch_rows)
                        # make sure we top up our in-flight rows before yielding
                        self._dispatch_input_rows()
                        self._log_state(f'yielding {len(batch_rows)} rows')
                        yield DataRowBatch(row_builder=self.row_builder, rows=batch_rows)
                        # at this point, we may have more completed rows

                assert self.completed_rows.empty()  # all completed rows should be sitting in output_buffer
                self.completed_event.clear()
                if self.input_complete and self.num_in_flight == 0:
                    # there is no more input and nothing left to wait for
                    assert self.avail_input_rows == 0
                    if self.output_buffer.num_ready > 0:
                        assert self.output_buffer.num_rows == self.output_buffer.num_ready
                        # yield the leftover rows
                        batch_rows = self.output_buffer.get_rows(self.output_buffer.num_ready)
                        self.num_output_rows += len(batch_rows)
                        self._log_state(f'yielding {len(batch_rows)} rows')
                        yield DataRowBatch(row_builder=self.row_builder, rows=batch_rows)

                    assert self.output_buffer.num_rows == 0
                    return

                if self.input_complete and self.avail_input_rows == 0 and not closed_evaluators:
                    # no more input rows to dispatch, but we're still waiting for rows to finish:
                    # close  all slot evaluators to flush queued rows
                    for evaluator in self.eval_ctx.slot_evaluators.values():
                        evaluator.close()
                    closed_evaluators = True

                # we don't have a full batch of rows at this point and need to wait
                aws = {exc_event_aw}  # always wait for an exception
                if self.next_input_batch is None and not self.input_complete:
                    # also wait for another batch if we don't have a read-ahead batch yet
                    if input_batch_aw is None:
                        input_batch_aw = asyncio.create_task(self._fetch_input_batch(), name='_fetch_input_batch()')
                    aws.add(input_batch_aw)
                if self.num_in_flight > 0:
                    # also wait for more rows to complete
                    if completed_aw is None:
                        completed_aw = asyncio.create_task(self.completed_event.wait(), name='completed.wait()')
                    aws.add(completed_aw)
                done, _ = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)

                if self.exc_event.is_set():
                    # we got an exception that we need to propagate through __iter__()
                    if isinstance(self.error, excs.ExprEvalError):
                        raise self.error from self.error.exc
                    else:
                        raise self.error
                if completed_aw in done:
                    self._log_state('completed_aw done')
                    completed_aw = None
                if input_batch_aw in done:
                    self._dispatch_input_rows()
                    input_batch_aw = None

        finally:
            # task cleanup
            active_tasks = {exc_event_aw}
            if input_batch_aw is not None:
                active_tasks.add(input_batch_aw)
            if completed_aw is not None:
                active_tasks.add(completed_aw)
            active_tasks.update(self.tasks)
            for task in active_tasks:
                if not task.done():
                    task.cancel()
            _ = await asyncio.gather(*active_tasks, return_exceptions=True)

            # expr cleanup
            exprs.Expr.release_list(self.eval_ctx.all_exprs)

    def dispatch_exc(
        self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType, exec_ctx: ExprEvalCtx
    ) -> None:
        """Propagate exception to main event loop or to dependent slots, depending on ignore_errors"""
        if len(rows) == 0 or self.exc_event.is_set():
            return

        if not self.ctx.ignore_errors:
            dependency_idxs = [e.slot_idx for e in exec_ctx.row_builder.unique_exprs[slot_with_exc].dependencies()]
            first_row = rows[0]
            input_vals = [first_row[idx] for idx in dependency_idxs]
            e = exec_ctx.row_builder.unique_exprs[slot_with_exc]
            self.error = excs.ExprEvalError(e, f'expression {e}', first_row.get_exc(e.slot_idx), exc_tb, input_vals, 0)
            self.exc_event.set()
            return

        for row in rows:
            assert row.has_exc(slot_with_exc)
            exc = row.get_exc(slot_with_exc)
            # propagate exception
            for slot_idx in np.nonzero(exec_ctx.row_builder.transitive_dependents[slot_with_exc])[0].tolist():
                row.set_exc(slot_idx, exc)
        self.dispatch(rows, exec_ctx)

    def dispatch(self, rows: list[exprs.DataRow], exec_ctx: ExprEvalCtx) -> None:
        """Dispatch rows to slot evaluators, based on materialized dependencies

        Uses 2D numpy arrays for vectorized operations across all rows.
        """
        if len(rows) == 0 or self.exc_event.is_set():
            return

        num_rows = len(rows)
        num_slots = exec_ctx.row_builder.num_materialized

        # Static arrays from row_builder
        dependencies = exec_ctx.row_builder.dependencies  # (num_slots, num_slots)
        num_dependencies = exec_ctx.row_builder.num_dependencies  # (num_slots,)

        # Stack row arrays into 2D arrays for vectorized operations
        # Shape: (num_rows, num_slots)
        has_val = np.stack([r.has_val for r in rows], axis=0)
        missing_slots = np.stack([r.missing_slots for r in rows], axis=0)
        is_scheduled = np.stack([r.is_scheduled for r in rows], axis=0)
        missing_dependents = np.stack([r.missing_dependents for r in rows], axis=0)

        # ---------- Compute progress (output slot materialization) ----------
        num_computed_outputs = 0
        if self.eval_ctx is exec_ctx:
            # Count newly-materialized output slots (before updating missing_slots)
            outputs_2d = self.outputs[np.newaxis, :]  # broadcast to (1, num_slots)
            missing_outputs_before = (missing_slots & outputs_2d).sum()

            # Update missing_slots: clear slots that now have values
            missing_slots &= (has_val == False)

            missing_outputs_after = (missing_slots & outputs_2d).sum()
            num_computed_outputs = int(missing_outputs_before - missing_outputs_after)
        else:
            # Update missing_slots: clear slots that now have values
            missing_slots &= (has_val == False)

        # ---------- Identify completed rows ----------
        missing_slot_counts = missing_slots.sum(axis=1)  # (num_rows,)
        completed_mask = (missing_slot_counts == 0)

        # ---------- Compute ready slots for non-completed rows ----------
        # ready_slots: (num_rows, num_slots)
        ready_slots = np.zeros((num_rows, num_slots), dtype=bool)

        non_completed_mask = ~completed_mask
        if np.any(non_completed_mask):
            # Only compute for non-completed rows
            nc_missing_slots = missing_slots[non_completed_mask]
            nc_has_val = has_val[non_completed_mask]
            nc_is_scheduled = is_scheduled[non_completed_mask]

            # missing_dependencies: dependencies needed for missing slots
            # Shape: (num_nc_rows, num_slots)
            nc_missing_dependencies = num_dependencies[np.newaxis, :] * nc_missing_slots

            # num_mat_dependencies: how many dependencies are materialized
            # has_val @ dependencies.T gives count of materialized dependencies per slot
            # dependencies[i, j] == True means expr i depends on expr j
            # We want: for each slot i, count how many of its dependencies have values
            nc_num_mat_dependencies = nc_has_val.astype(np.int32) @ dependencies.T.astype(np.int32)

            # Ready when all dependencies are materialized
            nc_num_missing = nc_missing_dependencies - nc_num_mat_dependencies
            nc_ready = (nc_num_missing == 0) & (nc_is_scheduled == False) & nc_missing_slots

            ready_slots[non_completed_mask] = nc_ready

            # Update is_scheduled for non-completed rows
            is_scheduled[non_completed_mask] |= nc_ready

        # ---------- GC computation ----------
        # Compute new missing_dependents for all rows
        # missing_dependents[i, slot] = count of slots that depend on 'slot' and don't have a value yet
        # dependencies[i, j] means expr i depends on expr j
        # For each slot j, we count how many slots i (that don't have values) depend on j
        # This is: (~has_val) @ dependencies, summing over the first axis
        new_missing_dependents = (~has_val).astype(np.int16) @ dependencies.astype(np.int16)

        # Identify slots to garbage collect per row
        gc_targets_2d = (new_missing_dependents == 0) & (missing_dependents > 0) & exec_ctx.gc_targets[np.newaxis, :]

        # ---------- Write back to DataRows and perform GC ----------
        for i, row in enumerate(rows):
            row.missing_slots = missing_slots[i]
            row.is_scheduled = is_scheduled[i]

            # Perform GC clear for slots that are no longer needed
            if gc_targets_2d[i].any():
                row.clear(gc_targets_2d[i])
            row.missing_dependents = new_missing_dependents[i]

        # ---------- Progress reporting ----------
        if self.progress_reporter is not None and num_computed_outputs > 0:
            self.progress_reporter.update(num_computed_outputs)

        # ---------- Handle completed rows ----------
        if np.any(completed_mask):
            completed_idxs = list(completed_mask.nonzero()[0])
            if rows[0].parent_row is not None:
                # these are nested rows
                for i in completed_idxs:
                    row = rows[i]
                    assert row.parent_row is not None and row.parent_slot_idx is not None
                    assert isinstance(row.parent_row.vals[row.parent_slot_idx], NestedRowList)
                    row.parent_row.vals[row.parent_slot_idx].complete_row()
            else:
                for i in completed_idxs:
                    self.completed_rows.put_nowait(rows[i])
                self.completed_event.set()
                self.num_in_flight -= len(completed_idxs)

        # ---------- Schedule ready slots ----------
        # Find slots that have any ready rows
        slots_with_ready_rows = ready_slots.sum(axis=0).nonzero()[0]
        for slot_idx in slots_with_ready_rows:
            ready_row_idxs = ready_slots[:, slot_idx].nonzero()[0]
            ready_rows = [rows[i] for i in ready_row_idxs]
            _logger.debug(f'Scheduling {len(ready_rows)} rows for slot {slot_idx}')
            exec_ctx.slot_evaluators[slot_idx].schedule(ready_rows, slot_idx)

    def register_task(self, t: asyncio.Task) -> None:
        self.tasks.add(t)
        t.add_done_callback(self._done_cb)

    def _done_cb(self, t: asyncio.Task) -> None:
        self.tasks.discard(t)
        # end the main loop if we had an unhandled exception
        try:
            t.result()
        except KeyboardInterrupt:
            # ExprEvalNode instances are long-running and reused across multiple operations.
            # When a user interrupts an operation (Ctrl+C), the main evaluation loop properly
            # handles the KeyboardInterrupt and terminates the current operation. However,
            # background tasks spawned by evaluators may complete asynchronously after the
            # operation has ended, and their done callbacks will fire during subsequent
            # operations. These "phantom" KeyboardInterrupt exceptions from previous
            # operations' background tasks should not interfere with new operations, so we
            # absorb them here rather than propagating them via self.error/self.exc_event.
            _logger.debug('Task completed with KeyboardInterrupt (user cancellation)')
            pass
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            stack_trace = traceback.format_exc()
            self.error = excs.Error(f'Exception in task: {exc}\n{stack_trace}')
            self.exc_event.set()

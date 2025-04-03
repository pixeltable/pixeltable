from __future__ import annotations

import asyncio
import datetime
import itertools
import logging
import sys
from typing import Any, Callable, Iterator, Optional, cast

import numpy as np

from pixeltable import exprs, func

from .globals import Dispatcher, Evaluator, FnCallArgs, ExecCtx

_logger = logging.getLogger('pixeltable')


class DefaultExprEvaluator(Evaluator):
    """
    Standard expression evaluation using Expr.eval().

    Creates one task per set of rows handed to schedule().

    TODO:
    - parallelize via Ray
    """

    e: exprs.Expr

    def __init__(self, e: exprs.Expr, dispatcher: Dispatcher, exec_ctx: ExecCtx):
        super().__init__(dispatcher, exec_ctx)
        self.e = e

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.e.slot_idx >= 0
        task = asyncio.create_task(self.eval(rows))
        self.dispatcher.register_task(task)

    async def eval(self, rows: list[exprs.DataRow]) -> None:
        rows_with_excs: set[int] = set()  # records idxs into rows
        for idx, row in enumerate(rows):
            assert not row.has_val[self.e.slot_idx] and not row.has_exc(self.e.slot_idx)
            if asyncio.current_task().cancelled() or self.dispatcher.exc_event.is_set():
                return
            try:
                self.e.eval(row, self.dispatcher.row_builder)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                row.set_exc(self.e.slot_idx, exc)
                rows_with_excs.add(idx)
                self.dispatcher.dispatch_exc([row], self.e.slot_idx, exc_tb, self.exec_ctx)
        self.dispatcher.dispatch([rows[i] for i in range(len(rows)) if i not in rows_with_excs], self.exec_ctx)


class FnCallEvaluator(Evaluator):
    """
    Evaluates function calls:
    - batched functions (sync and async): one task per batch
    - async functions: one task per row
    - the rest: one task per set of rows handed to schedule()

    TODO:
    - adaptive batching: finding the optimal batch size based on observed execution times
    """

    fn_call: exprs.FunctionCall
    fn: func.CallableFunction
    scalar_py_fn: Optional[Callable]  # only set for non-batching CallableFunctions

    # only set if fn.is_batched
    call_args_queue: Optional[asyncio.Queue[FnCallArgs]]  # FnCallArgs waiting for execution
    batch_size: Optional[int]

    def __init__(self, fn_call: exprs.FunctionCall, dispatcher: Dispatcher, exec_ctx: ExecCtx):
        super().__init__(dispatcher, exec_ctx)
        self.fn_call = fn_call
        self.fn = cast(func.CallableFunction, fn_call.fn)
        if isinstance(self.fn, func.CallableFunction) and self.fn.is_batched:
            self.call_args_queue = asyncio.Queue[FnCallArgs]()
            # we're not supplying sample arguments there, they're ignored anyway
            self.batch_size = self.fn.get_batch_size()
            self.scalar_py_fn = None
        else:
            self.call_args_queue = None
            self.batch_size = None
            if isinstance(self.fn, func.CallableFunction):
                self.scalar_py_fn = self.fn.py_fn
            else:
                self.scalar_py_fn = None

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.fn_call.slot_idx >= 0

        # create FnCallArgs for incoming rows
        skip_rows: list[exprs.DataRow] = []  # skip rows with Nones in non-nullable parameters
        rows_call_args: list[FnCallArgs] = []
        for row in rows:
            args_kwargs = self.fn_call.make_args(row)
            if args_kwargs is None:
                # nothing to do here
                row[self.fn_call.slot_idx] = None
                skip_rows.append(row)
            else:
                args, kwargs = args_kwargs
                rows_call_args.append(FnCallArgs(self.fn_call, [row], args=args, kwargs=kwargs))

        if len(skip_rows) > 0:
            self.dispatcher.dispatch(skip_rows, self.exec_ctx)

        if self.batch_size is not None:
            if not self.is_closed and (len(rows_call_args) + self.call_args_queue.qsize() < self.batch_size):
                # we don't have enough FnCallArgs for a batch, so add them to the queue
                for item in rows_call_args:
                    self.call_args_queue.put_nowait(item)
                return

            # create one task per batch
            combined_call_args = itertools.chain(self._queued_call_args_iter(), rows_call_args)
            while True:
                call_args_batch = list(itertools.islice(combined_call_args, self.batch_size))
                if len(call_args_batch) == 0:
                    break
                if len(call_args_batch) < self.batch_size and not self.is_closed:
                    # we don't have a full batch left: return the rest to the queue
                    assert self.call_args_queue.empty()  # we saw all queued items
                    for item in call_args_batch:
                        self.call_args_queue.put_nowait(item)
                    return

                # turn call_args_batch into a single batched FnCallArgs
                _logger.debug(f'Creating batch of size {len(call_args_batch)} for slot {slot_idx}')
                batched_call_args = self._create_batch_call_args(call_args_batch)
                if self.fn_call.resource_pool is not None:
                    # hand the call off to the resource pool's scheduler
                    scheduler = self.dispatcher.schedulers[self.fn_call.resource_pool]
                    scheduler.submit(batched_call_args)
                else:
                    task = asyncio.create_task(self.eval_batch(batched_call_args))
                    self.dispatcher.register_task(task)

        elif self.fn.is_async:
            if self.fn_call.resource_pool is not None:
                # hand the call off to the resource pool's scheduler
                scheduler = self.dispatcher.schedulers[self.fn_call.resource_pool]
                for item in rows_call_args:
                    scheduler.submit(item)
            else:
                # create one task per call
                for item in rows_call_args:
                    task = asyncio.create_task(self.eval_async(item))
                    self.dispatcher.register_task(task)

        else:
            # create a single task for all rows
            task = asyncio.create_task(self.eval(rows_call_args))
            self.dispatcher.register_task(task)

    def _queued_call_args_iter(self) -> Iterator[FnCallArgs]:
        while not self.call_args_queue.empty():
            yield self.call_args_queue.get_nowait()

    def _create_batch_call_args(self, call_args: list[FnCallArgs]) -> FnCallArgs:
        """Roll call_args into a single batched FnCallArgs"""
        batch_args: list[list[Optional[Any]]] = [[None] * len(call_args) for _ in range(len(self.fn_call.arg_idxs))]
        batch_kwargs: dict[str, list[Optional[Any]]] = {
            k: [None] * len(call_args) for k in self.fn_call.kwarg_idxs.keys()
        }
        assert isinstance(self.fn, func.CallableFunction)
        for i, item in enumerate(call_args):
            for j in range(len(item.args)):
                batch_args[j][i] = item.args[j]
            for k in item.kwargs.keys():
                batch_kwargs[k][i] = item.kwargs[k]
        return FnCallArgs(
            self.fn_call, [item.row for item in call_args], batch_args=batch_args, batch_kwargs=batch_kwargs
        )

    async def eval_batch(self, batched_call_args: FnCallArgs) -> None:
        result_batch: list[Any]
        try:
            if self.fn.is_async:
                result_batch = await self.fn.aexec_batch(
                    *batched_call_args.batch_args, **batched_call_args.batch_kwargs
                )
            else:
                # check for cancellation before starting something potentially long-running
                if asyncio.current_task().cancelled() or self.dispatcher.exc_event.is_set():
                    return
                result_batch = self.fn.exec_batch(batched_call_args.batch_args, batched_call_args.batch_kwargs)
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            for row in batched_call_args.rows:
                row.set_exc(self.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(batched_call_args.rows, self.fn_call.slot_idx, exc_tb, self.exec_ctx)
            return

        for i, row in enumerate(batched_call_args.rows):
            row[self.fn_call.slot_idx] = result_batch[i]
        self.dispatcher.dispatch(batched_call_args.rows, self.exec_ctx)

    async def eval_async(self, call_args: FnCallArgs) -> None:
        assert len(call_args.rows) == 1
        assert not call_args.row.has_val[self.fn_call.slot_idx]
        assert not call_args.row.has_exc(self.fn_call.slot_idx)

        try:
            start_ts = datetime.datetime.now()
            _logger.debug(f'Start evaluating slot {self.fn_call.slot_idx}')
            call_args.row[self.fn_call.slot_idx] = await self.fn.aexec(*call_args.args, **call_args.kwargs)
            end_ts = datetime.datetime.now()
            _logger.debug(f'Evaluated slot {self.fn_call.slot_idx} in {end_ts - start_ts}')
            self.dispatcher.dispatch([call_args.row], self.exec_ctx)
        except Exception as exc:
            import anthropic

            if isinstance(exc, anthropic.RateLimitError):
                _logger.debug(f'RateLimitError: {exc}')
            _, _, exc_tb = sys.exc_info()
            call_args.row.set_exc(self.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(call_args.rows, self.fn_call.slot_idx, exc_tb, self.exec_ctx)

    async def eval(self, call_args_batch: list[FnCallArgs]) -> None:
        rows_with_excs: set[int] = set()  # records idxs into 'rows'
        for idx, item in enumerate(call_args_batch):
            assert len(item.rows) == 1
            assert not item.row.has_val[self.fn_call.slot_idx]
            assert not item.row.has_exc(self.fn_call.slot_idx)
            # check for cancellation before starting something potentially long-running
            if asyncio.current_task().cancelled() or self.dispatcher.exc_event.is_set():
                return
            try:
                item.row[self.fn_call.slot_idx] = self.scalar_py_fn(*item.args, **item.kwargs)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                item.row.set_exc(self.fn_call.slot_idx, exc)
                rows_with_excs.add(idx)
                self.dispatcher.dispatch_exc(item.rows, self.fn_call.slot_idx, exc_tb, self.exec_ctx)
        self.dispatcher.dispatch(
            [call_args_batch[i].row for i in range(len(call_args_batch)) if i not in rows_with_excs], self.exec_ctx
        )

    def _close(self) -> None:
        """Create a task for the incomplete batch of queued FnCallArgs, if any"""
        _logger.debug(f'FnCallEvaluator.close(): slot_idx={self.fn_call.slot_idx}')
        if self.call_args_queue is None or self.call_args_queue.empty():
            return
        batched_call_args = self._create_batch_call_args(list(self._queued_call_args_iter()))
        task = asyncio.create_task(self.eval_batch(batched_call_args))
        self.dispatcher.register_task(task)


class NestedRowList:
    """ """

    rows: list[exprs.DataRow]
    num_completed: int
    completion: asyncio.Event

    def __init__(self, rows: list[exprs.DataRow]):
        self.num_completed = 0
        self.rows = rows
        self.completion = asyncio.Event()

    def complete_row(self) -> None:
        self.num_completed += 1
        if self.num_completed == len(self.rows):
            self.completion.set()


class JsonMapperDispatcher(Evaluator):
    """ """

    e: exprs.JsonMapperDispatch
    target_expr: exprs.Expr
    scope_anchor: exprs.ObjectRef
    nested_exec_ctx: ExecCtx
    #nested_row_builder: exprs.RowBuilder
    #target_expr_eval_ctx: np.ndarray  # ExprEvalCtx.slot_idxs as an ndarray

    def __init__(self, e: exprs.JsonMapperDispatch, dispatcher: Dispatcher, exec_ctx: ExecCtx):
        super().__init__(dispatcher, exec_ctx)
        self.e = e
        self.target_expr = e.target_expr.copy()  # we need new slot idxs
        self.scope_anchor = e.scope_anchor.copy()
        nested_row_builder = exprs.RowBuilder(output_exprs=[self.target_expr], columns=[], input_exprs=[])
        nested_row_builder.set_slot_idxs([self.target_expr, self.scope_anchor])
        self.nested_exec_ctx = ExecCtx(dispatcher, nested_row_builder, [self.target_expr], [])

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.e.slot_idx >= 0
        for row in rows:
            src = row[self.e.src_expr.slot_idx]
            if not isinstance(src, list):
                # invalid/non-list src path
                row[self.e.slot_idx] = None
                continue

            nested_rows = [
                exprs.DataRow(
                    size=self.nested_exec_ctx.row_builder.num_materialized,
                    img_slot_idxs=[],
                    media_slot_idxs=[],
                    array_slot_idxs=[],
                    parent_row=row,
                    parent_slot_idx=self.e.slot_idx,
                )
                for _ in src
            ]
            for nested_row, anchor_val in zip(nested_rows, src):
                nested_row[self.scope_anchor.slot_idx] = anchor_val
                nested_row.missing_slots = self.nested_exec_ctx.eval_ctx & (nested_row.has_val == False)
                nested_row.missing_dependents = np.sum(
                    self.nested_exec_ctx.row_builder.dependencies[nested_row.has_val == False], axis=0
                )
            row.vals[self.e.slot_idx] = NestedRowList(nested_rows)
            self.dispatcher.dispatch(nested_rows, self.nested_exec_ctx)

        task = asyncio.create_task(self.gather(rows))
        self.dispatcher.register_task(task)

    async def gather(self, rows: list[exprs.DataRow]) -> None:
        remove_me = 1
        for row in rows:
            assert not row.has_val[self.e.slot_idx]
            assert isinstance(row.vals[self.e.slot_idx], NestedRowList)
            await row[self.e.slot_idx].completion.wait()
            row.has_val[self.e.slot_idx] = True
        self.dispatcher.dispatch(rows, self.exec_ctx)

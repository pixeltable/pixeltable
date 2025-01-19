from __future__ import annotations

import asyncio
import datetime
import itertools
import logging
import sys
from typing import Iterator, Any, Optional, Callable

from pixeltable import exprs
from pixeltable import func
from .globals import Dispatcher, Evaluator, FnCallArgs

_logger = logging.getLogger('pixeltable')


class DefaultExprEvaluator(Evaluator):
    """
    Standard expression evaluation using Expr.eval().

    Creates one task per set of rows handed to schedule().

    TODO:
    - parallelize via Ray
    """
    e: exprs.Expr

    def __init__(self, e: exprs.Expr, dispatcher: Dispatcher):
        super().__init__(dispatcher)
        self.e = e

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.e.slot_idx >= 0
        task = asyncio.create_task(self.eval(rows))
        self.dispatcher.tasks.add(task)
        task.add_done_callback(self.dispatcher.tasks.discard)

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
                self.dispatcher.dispatch_exc([row], self.e.slot_idx, exc_tb)
        self.dispatcher.dispatch([rows[i] for i in range(len(rows)) if i not in rows_with_excs])


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
    fn: func.Function
    scalar_py_fn: Optional[Callable]  # only set for non-batching CallableFunctions

    # only set if fn.is_batched
    input_queue: Optional[asyncio.Queue[FnCallArgs]]
    batch_size: Optional[int]

    def __init__(self, fn_call: exprs.FunctionCall, dispatcher: Dispatcher):
        super().__init__(dispatcher)
        self.fn_call = fn_call
        self.fn = fn_call.fn
        if isinstance(self.fn, func.CallableFunction) and self.fn.is_batched:
            self.input_queue =  asyncio.Queue[FnCallArgs]()
            # we're not supplying sample arguments there, they're ignored anyway
            self.batch_size = self.fn.get_batch_size()
            self.scalar_py_fn = None
        else:
            self.input_queue = None
            self.batch_size = None
            if isinstance(self.fn, func.CallableFunction):
                self.scalar_py_fn = self.fn.py_fn
            else:
                self.scalar_py_fn = None

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.fn_call.slot_idx >= 0

        # create FnCallArgs for incoming rows
        skip_rows: list[exprs.DataRow] = []  # skip rows with Nones in non-nullable parameters
        queue_items: list[FnCallArgs] = []
        for row in rows:
            args_kwargs = self.fn_call.make_args(row)
            if args_kwargs is None:
                # nothing to do here
                row[self.fn_call.slot_idx] = None
                skip_rows.append(row)
            else:
                args, kwargs = args_kwargs
                queue_items.append(FnCallArgs(self.fn_call, row, args, kwargs))

        if len(skip_rows) > 0:
            self.dispatcher.dispatch(skip_rows)

        if self.batch_size is not None:
            if not self.is_closed and (len(queue_items) + self.input_queue.qsize() < self.batch_size):
                # we don't have enough rows for a batch
                for item in queue_items:
                    self.input_queue.put_nowait(item)
                return

            # create one task per batch
            combined_items = itertools.chain(self._queued_items(), queue_items)
            while True:
                batch_items = list(itertools.islice(combined_items, self.batch_size))
                if len(batch_items) == 0:
                    break
                if len(batch_items) < self.batch_size and not self.is_closed:
                    # we don't have a full batch left: return the rest to the queue
                    assert self.input_queue.empty()  # we saw all queued items
                    for item in batch_items:
                        self.input_queue.put_nowait(item)
                    return
                task = asyncio.create_task(self.eval_batch(batch_items))
                self.dispatcher.tasks.add(task)
                task.add_done_callback(self.dispatcher.tasks.discard)
        elif self.fn.is_async:
            if self.fn_call.resource_pool is not None:
                scheduler = self.dispatcher.schedulers[self.fn_call.resource_pool]
                for item in queue_items:
                    scheduler.submit(item)
            else:
                # create one task per call
                for item in queue_items:
                    task = asyncio.create_task(self.eval_async(item))
                    self.dispatcher.tasks.add(task)
                    task.add_done_callback(self.dispatcher.tasks.discard)

        else:
            # create a single task for all rows
            task = asyncio.create_task(self.eval(queue_items))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.tasks.discard)

    def _queued_items(self) -> Iterator[FnCallArgs]:
        while not self.input_queue.empty():
            yield self.input_queue.get_nowait()

    async def eval_batch(self, items: list[FnCallArgs]) -> None:
        arg_batches: list[list[Optional[Any]]] = [[None] * len(items) for _ in range(len(self.fn_call.args))]
        kwarg_batches: dict[str, list[Optional[Any]]] = {k: [None] * len(items) for k in self.fn_call.kwargs.keys()}

        assert isinstance(self.fn, func.CallableFunction)
        for i, item in enumerate(items):
            for j in range(len(item.args)):
                arg_batches[j][i] = item.args[j]
            for k in item.kwargs.keys():
                kwarg_batches[k][i] = item.kwargs[k]
        rows = [item.row for item in items]
        result_batch: list[Any]
        try:
            if self.fn.is_async:
                result_batch = await self.fn.aexec_batch(*arg_batches, **kwarg_batches)
            else:
                result_batch = self.fn.exec_batch(arg_batches, kwarg_batches)
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            for row in rows:
                row.set_exc(self.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(rows, self.fn_call.slot_idx, exc_tb)
            return

        for i, row in enumerate(rows):
            row[self.fn_call.slot_idx] = result_batch[i]
        self.dispatcher.dispatch(rows)

    async def eval_async(self, item: FnCallArgs) -> None:
        assert not item.row.has_val[self.fn_call.slot_idx] and not item.row.has_exc(self.fn_call.slot_idx)

        try:
            start_ts = datetime.datetime.now()
            _logger.debug(f'Start evaluating slot {self.fn_call.slot_idx}')
            item.row[self.fn_call.slot_idx] = await self.fn.aexec(*item.args, **item.kwargs)
            end_ts = datetime.datetime.now()
            _logger.debug(f'Evaluated slot {self.fn_call.slot_idx} in {end_ts - start_ts}')
            self.dispatcher.dispatch([item.row])
        except Exception as exc:
            import anthropic
            if isinstance(exc, anthropic.RateLimitError):
                _logger.debug(f'RateLimitError: {exc}')
            _, _, exc_tb = sys.exc_info()
            item.row.set_exc(self.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc([item.row], self.fn_call.slot_idx, exc_tb)

    async def eval(self, items: list[FnCallArgs]) -> None:
        rows_with_excs: set[int] = set()  # records idxs into 'rows'
        for idx, item in enumerate(items):
            assert not item.row.has_val[self.fn_call.slot_idx] and not item.row.has_exc(self.fn_call.slot_idx)
            if asyncio.current_task().cancelled() or self.dispatcher.exc_event.is_set():
                return
            try:
                item.row[self.fn_call.slot_idx] = self.scalar_py_fn(*item.args, **item.kwargs)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                item.row.set_exc(self.fn_call.slot_idx, exc)
                rows_with_excs.add(idx)
                self.dispatcher.dispatch_exc([item.row], self.fn_call.slot_idx, exc_tb)
        self.dispatcher.dispatch([items[i].row for i in range(len(items)) if i not in rows_with_excs])

    def _close(self) -> None:
        """Create a task for the remaining queued items"""
        if self.input_queue is None or self.input_queue.empty():
            return
        task = asyncio.create_task(self.eval_batch(list(self._queued_items())))
        self.dispatcher.tasks.add(task)
        task.add_done_callback(self.dispatcher.tasks.discard)


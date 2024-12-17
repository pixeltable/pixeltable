import abc
import asyncio
import dataclasses
import itertools
import logging
import sys
from typing import Iterable, Iterator, Any, AsyncIterator, Protocol, Optional
from types import TracebackType
import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pixeltable.exceptions as excs
from pixeltable import exprs
from pixeltable import func
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class Dispatcher(Protocol):
    tasks: set[asyncio.Task]
    row_builder: exprs.RowBuilder
    exc_event: asyncio.Event

    def dispatch(self, rows: list[exprs.DataRow]) -> None:
        """Dispatches row slots to the appropriate schedulers; does not block"""
        ...

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        """Propagates exception in slot_with_exc to all dependent slots and dispatches the rest; does not block"""
        ...


class Evaluator(abc.ABC):
    dispatcher: Dispatcher
    is_closed: bool

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.is_closed = False

    @abc.abstractmethod
    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        """Create tasks to evaluate the expression in the given slot for the given rows; must not block"""
        pass

    def _close(self) -> None:
        """Close the evaluator; must not block"""
        pass

    def close(self) -> None:
        self.is_closed = True
        self._close()


class DefaultExprEvaluator(Evaluator):
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
        rows_with_excs: set[int] = set()  # records idxs into 'rows'
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


class QueryTemplateFnCallExecutor(Evaluator):
    fn_call: exprs.FunctionCall
    query_template_fn: func.QueryTemplateFunction

    def __init__(self, fn_call: exprs.FunctionCall, dispatcher: Dispatcher):
        super().__init__(dispatcher)
        self.fn_call = fn_call
        assert isinstance(fn_call.fn, func.QueryTemplateFunction)
        self.query_template_fn = fn_call.fn

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.fn_call.slot_idx >= 0
        task = asyncio.create_task(self.eval(rows))
        self.dispatcher.tasks.add(task)
        task.add_done_callback(self.dispatcher.tasks.discard)

    async def eval(self, rows: list[exprs.DataRow]) -> None:
        rows_with_excs: set[int] = set()  # records idxs into 'rows'
        for idx, row in enumerate(rows):
            assert not row.has_val[self.fn_call.slot_idx] and not row.has_exc(self.fn_call.slot_idx)
            # TODO: is it possible for this to get cancelled after it got started?
            if asyncio.current_task().cancelled():
                return
            try:
                args_kwargs = self.fn_call.make_args(row)
                if args_kwargs is None:
                    # we can't evaluate this function
                    row[self.fn_call.slot_idx] = None
                else:
                    args, kwargs = args_kwargs
                    row[self.fn_call.slot_idx] = await self.query_template_fn.aexec(*args, **kwargs)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                row.set_exc(self.fn_call.slot_idx, exc)
                rows_with_excs.add(idx)
                self.dispatcher.dispatch_exc([row], self.fn_call.slot_idx, exc_tb)
        self.dispatcher.dispatch([rows[i] for i in range(len(rows)) if i not in rows_with_excs])


class BatchedFnCallEvaluator(Evaluator):
    @dataclasses.dataclass
    class QueueItem:
        row: exprs.DataRow
        args: list[Any]
        kwargs: dict[str, Any]

    fn_call: exprs.FunctionCall
    input_queue: asyncio.Queue[QueueItem]
    batch_size: int

    def __init__(self, fn_call: exprs.FunctionCall, dispatcher: Dispatcher):
        super().__init__(dispatcher)
        assert isinstance(fn_call.fn, func.CallableFunction)
        assert fn_call.fn.is_batched
        self.fn_call = fn_call
        self.input_queue =  asyncio.Queue[BatchedFnCallEvaluator.QueueItem]()
        # we're not supplying sample arguments there, they're ignored anyway
        self.batch_size = fn_call.fn.get_batch_size()

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.fn_call.slot_idx >= 0

        # create QueueItems for incoming rows
        skip_rows: list[exprs.DataRow] = []  # skip rows with Nones in non-nullable parameters
        queue_items: list[BatchedFnCallEvaluator.QueueItem] = []
        for row in rows:
            args_kwargs = self.fn_call.make_args(row)
            if args_kwargs is None:
                # nothing to do here
                row[self.fn_call.slot_idx] = None
                skip_rows.append(row)
            else:
                args, kwargs = args_kwargs
                queue_items.append(self.QueueItem(row, args, kwargs))

        if len(skip_rows) > 0:
            self.dispatcher.dispatch(skip_rows)

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
            if not self.is_closed and (len(batch_items) < self.batch_size):
                # we don't have a full batch left: return the rest to the queue
                assert self.input_queue.empty()  # we returned all queued items
                for item in batch_items:
                    self.input_queue.put_nowait(item)
                return
            task = asyncio.create_task(self.eval(batch_items))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.tasks.discard)

    def _queued_items(self) -> Iterator[QueueItem]:
        while not self.input_queue.empty():
            yield self.input_queue.get_nowait()

    async def eval(self, items: list[QueueItem]) -> None:
        arg_batches: list[list[Optional[Any]]] = [[None] * len(items) for _ in range(len(self.fn_call.args))]
        kwarg_batches: dict[str, list[Optional[Any]]] = {k: [None] * len(items) for k in self.fn_call.kwargs.keys()}

        assert isinstance(self.fn_call.fn, func.CallableFunction)
        for i, item in enumerate(items):
            for j in range(len(item.args)):
                arg_batches[j][i] = item.args[j]
            for k in item.kwargs.keys():
                kwarg_batches[k][i] = item.kwargs[k]
        rows = [item.row for item in items]
        try:
            result_batch = self.fn_call.fn.exec_batch(*arg_batches, **kwarg_batches)
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            for row in rows:
                row.set_exc(self.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(rows, self.fn_call.slot_idx, exc_tb)
            return

        for i, row in enumerate(rows):
            row[self.fn_call.slot_idx] = result_batch[i]
        self.dispatcher.dispatch(rows)

    def _close(self) -> None:
        """Create a task for the remaining queued items"""
        if self.input_queue.empty():
            return
        task = asyncio.create_task(self.eval(list(self._queued_items())))
        self.dispatcher.tasks.add(task)
        task.add_done_callback(self.dispatcher.tasks.discard)


class AsyncFnCallEvaluator(Evaluator):
    @dataclasses.dataclass
    class QueueItem:
        row: exprs.DataRow
        args: list[Any]
        kwargs: dict[str, Any]

    fn_call: exprs.FunctionCall
    fn: func.CallableFunction

    def __init__(self, fn_call: exprs.FunctionCall, dispatcher: Dispatcher):
        super().__init__(dispatcher)
        assert isinstance(fn_call.fn, func.CallableFunction)
        self.fn_call = fn_call
        assert isinstance(fn_call.fn, func.CallableFunction)
        assert fn_call.fn.is_async
        self.fn = fn_call.fn

    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        assert self.fn_call.slot_idx >= 0
        for row in rows:
            task = asyncio.create_task(self.eval(row))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.tasks.discard)

    async def eval(self, row: exprs.DataRow) -> None:
        assert not row.has_val[self.fn_call.slot_idx] and not row.has_exc(self.fn_call.slot_idx)

        args_kwargs = self.fn_call.make_args(row)
        if args_kwargs is None:
            # we can't evaluate this function
            row[self.fn_call.slot_idx] = None
            self.dispatcher.dispatch([row])
            return

        args, kwargs = args_kwargs
        try:
            start_ts = datetime.datetime.now()
            _logger.debug(f'Start evaluating slot {self.fn_call.slot_idx}')
            row[self.fn_call.slot_idx] = await self.fn.async_exec(*args, **kwargs)
            end_ts = datetime.datetime.now()
            _logger.debug(f'Evaluated slot {self.fn_call.slot_idx} in {end_ts - start_ts}')
            self.dispatcher.dispatch([row])
        except Exception as exc:
            _, _, exc_tb = sys.exc_info()
            row.set_exc(self.fn_call.slot_idx, exc)
            if False:
                input_vals = [row[d.slot_idx] for d in self.e.dependencies()]
                raise excs.ExprEvalError(
                    self.e, f'expression {self.e}', row.get_exc(self.e.slot_idx), exc_tb, input_vals, 0)
            self.dispatcher.dispatch_exc([row], self.fn_call.slot_idx, exc_tb)


class CircularRowBuffer:
    size: int
    maintain_order: bool  # True if we're returning rows order by position
    num_rows: int  # number of rows in the buffer
    num_ready: int  # number of consecutive non-None rows at head
    buffer: np.ndarray  # of object
    head_idx: int  # index of beginning of the buffer
    head_row_pos: int  # row position of the beginning of the buffer

    def __init__(self, size: int, maintain_order: bool):
        self.size = size
        self.maintain_order = maintain_order
        self.num_rows = 0
        self.num_ready = 0
        self.buffer = np.full(size, None, dtype=object)
        self.head_row_pos = 0
        self.head_idx = 0

    def add_row(self, row: exprs.DataRow, pos: Optional[int]) -> None:
        assert pos is None or (pos - self.head_row_pos < self.size)
        offset: int  # of new row from head
        if self.maintain_order:
            offset = pos - self.head_row_pos
        else:
            offset = self.num_rows
        idx = (self.head_idx + offset) % self.size
        assert self.buffer[idx] is None

        self.buffer[idx] = row
        self.num_rows += 1
        if self.maintain_order:
            if offset == self.num_ready:
                # we have new ready rows; find out how many
                while offset < self.size and self.buffer[(self.head_idx + offset) % self.size] is not None:
                    offset += 1
                self.num_ready = offset
        else:
            self.num_ready += 1

    def get_rows(self, n: int) -> list[exprs.DataRow]:
        """Get up to n ready rows from head"""
        n = min(n, self.num_ready)
        if n == 0:
            return []
        rows: list[exprs.DataRow]
        if self.head_idx + n <= self.size:
            rows = self.buffer[self.head_idx:self.head_idx + n].tolist()
            self.buffer[self.head_idx:self.head_idx + n] = None
        else:
            rows = np.concatenate([self.buffer[self.head_idx:], self.buffer[:self.head_idx + n - self.size]]).tolist()
            self.buffer[self.head_idx:] = None
            self.buffer[:self.head_idx + n - self.size] = None
        self.head_row_pos += n
        self.head_idx = (self.head_idx + n) % self.size
        self.num_rows -= n
        self.num_ready -= n
        return rows


class AsyncExprEvalNode(ExecNode):
    """
    TODO:
    - flush in-mem images of output columns when they're no longer needed
    """
    maintain_input_order: bool  # True if we're returning rows in the order we received them from our input
    num_dependencies: np.ndarray  # number of dependencies for our output slots; indexed by slot idx
    output_mask: np.ndarray  # of bool; True if this slot is part of our output
    slot_evaluators: dict[int, Evaluator]
    tasks: set[asyncio.Task]
    completed_rows: asyncio.Queue[exprs.DataRow]  # rows ready to be returned
    gc_targets: np.ndarray  # bool per slot; True if this is an intermediate expr (ie, not part of our output)
    eval_ctx: np.ndarray  # bool per slot; EvalCtx.slot_idxs as a mask
    exc_event: asyncio.Event  # set if an exception needs to be propagated
    eval_error: Optional[excs.ExprEvalError]  # set if an error needs to be propagated

    # debugging
    num_input_rows: int
    num_output_rows: int

    BATCH_SIZE = 12
    MAX_IN_FLIGHT = 50

    def __init__(
        self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr], input_exprs: Iterable[exprs.Expr],
        maintain_input_order: bool, input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.maintain_input_order = maintain_input_order
        self.num_dependencies = np.sum(row_builder.dependencies, axis=1)
        self.output_mask = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.output_mask[output_slot_idxs] = True
        self.tasks = set()

        self.gc_targets = np.ones(row_builder.num_materialized, dtype=bool)
        # we need to retain all slots that are part of the output
        self.gc_targets[[e.slot_idx for e in row_builder.output_exprs]] = False

        output_ctx = self.row_builder.create_eval_ctx(output_exprs, exclude=input_exprs)
        self.eval_ctx = np.zeros(row_builder.num_materialized, dtype=bool)
        self.eval_ctx[output_ctx.slot_idxs] = True
        self.eval_error = None

        self.num_input_rows = 0
        self.num_output_rows = 0

        self.slot_evaluators = {}
        self._init_slot_evaluators()

    def _init_slot_evaluators(self) -> None:
        for slot_idx in range(self.row_builder.num_materialized):
            expr = self.row_builder.unique_exprs[slot_idx]
            if isinstance(expr, exprs.FunctionCall) and isinstance(expr.fn, func.CallableFunction):
                fn = expr.fn
                if fn.is_batched:
                    self.slot_evaluators[slot_idx] = BatchedFnCallEvaluator(expr, self)
                elif fn.is_async:
                    self.slot_evaluators[slot_idx] = AsyncFnCallEvaluator(expr, self)
                else:
                    self.slot_evaluators[slot_idx] = DefaultExprEvaluator(expr, self)
            elif isinstance(expr, exprs.FunctionCall) and isinstance(expr.fn, func.QueryTemplateFunction):
                self.slot_evaluators[slot_idx] = QueryTemplateFnCallExecutor(expr, self)
            else:
                self.slot_evaluators[slot_idx] = DefaultExprEvaluator(expr, self)

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        self.completed_rows = asyncio.Queue[exprs.DataRow]()
        self.exc_event = asyncio.Event()
        exc_event_wait = asyncio.create_task(self.exc_event.wait())
        #batch_rows: list[exprs.DataRow] = []
        input_row_iter = self._input_rows()
        input_complete = False
        num_in_flight = 0  # number of dispatched rows that haven't been returned yet
        row_pos = 0
        row_pos_map: dict[int, int] = {}  # id(row) -> row_pos
        output_buffer = CircularRowBuffer(self.MAX_IN_FLIGHT, maintain_order=self.maintain_input_order)

        _logger.debug('Starting AsyncExprEvalNode')
        while True:
            if not input_complete and num_in_flight < self.MAX_IN_FLIGHT:
                # get more input rows scheduled
                _logger.debug(f'Trying to fetch {self.MAX_IN_FLIGHT - num_in_flight} input rows')
                num_requested = self.MAX_IN_FLIGHT - num_in_flight
                input_rows: list[exprs.DataRow] = []
                try:
                    for _ in range(num_requested):
                        row = await input_row_iter.__anext__()
                        row_pos_map[id(row)] = row_pos
                        row_pos += 1
                        input_rows.append(row)
                except StopAsyncIteration:
                    _logger.debug('Input is complete')
                    input_complete = True
                    for evaluator in self.slot_evaluators.values():
                        evaluator.close()
                _logger.debug(f'Fetched {len(input_rows)} input rows')
                self.num_input_rows += len(input_rows)
                num_in_flight += len(input_rows)
                self._init_input_rows(input_rows)
                self.dispatch(input_rows)

            # try to assemble output batch
            #assert len(batch_rows) < self.BATCH_SIZE  # otherwise we would have yielded already
            #_logger.debug(f'Assembling batch; #batch_rows={len(batch_rows)}')
            while output_buffer.num_ready < self.BATCH_SIZE:
                _logger.debug(f'#completed_rows={self.completed_rows.qsize()}')
                row: exprs.DataRow
                if not self.completed_rows.empty():
                    row = self.completed_rows.get_nowait()
                    _logger.debug(f'Gathered completed row')
                else:
                    # there are no completed rows to add to the batch
                    if not input_complete and num_in_flight < self.MAX_IN_FLIGHT:
                        # we can add more in-flight rows; let's do that now
                        break
                    if input_complete and num_in_flight == 0:
                        # there's nothing left to wait for
                        break
                    # we need to wait for more rows to complete or for an exception
                    _logger.debug('Waiting for more rows to complete')
                    tasks = [
                        asyncio.create_task(self.completed_rows.get(), name='completed_rows.get()'),
                        exc_event_wait,
                    ]
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    if self.exc_event.is_set():
                        # we got an exception that we need to propagate through __iter__();
                        # cancel all in-flight tasks first
                        for task in itertools.chain(pending, self.tasks):
                            task.cancel()
                        _ = await asyncio.gather(*pending, *list(self.tasks), return_exceptions=True)
                        _logger.debug(f'Propagating exception {self.eval_error}')
                        raise self.eval_error
                    row_ = done.pop().result()
                    assert isinstance(row_, exprs.DataRow)
                    row = row_
                output_buffer.add_row(row, row_pos_map[id(row)])
                num_in_flight -= 1

            _logger.debug(f'Finished assembly; #batch_rows={output_buffer.num_ready}')
            if output_buffer.num_ready >= self.BATCH_SIZE or (input_complete and num_in_flight == 0):
                batch_rows = output_buffer.get_rows(self.BATCH_SIZE)
                _logger.debug(f'Yielding batch of size {len(batch_rows)}')
                self.num_output_rows += len(batch_rows)
                yield DataRowBatch(tbl=None, row_builder=self.row_builder, rows=batch_rows)

            if input_complete and num_in_flight == 0:
                #assert len(batch_rows) == 0
                assert output_buffer.num_rows == 0
                _logger.debug('returning from AsyncExprEvalNode')
                # clean up exc_event_wait
                exc_event_wait.cancel()
                try:
                    _ = await exc_event_wait
                except asyncio.CancelledError:
                    pass
                return

    def _init_input_rows(self, rows: list[exprs.DataRow]) -> None:
        """Set expr eval state in DataRow"""
        for row in rows:
            row.missing_dependents = np.sum(self.row_builder.dependencies[row.has_val == False], axis=0)
            row.missing_slots = self.eval_ctx & (row.has_val == False)

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        if len(rows) == 0 or self.exc_event.is_set():
            return

        if not self.ctx.ignore_errors:
            #dependency_idxs = np.nonzero(self.row_builder.dependencies[slot_with_exc])[0].tolist()
            dependency_idxs = [e.slot_idx for e in self.row_builder.unique_exprs[slot_with_exc].dependencies()]
            first_row = rows[0]
            input_vals = [first_row[idx] for idx in dependency_idxs]
            e = self.row_builder.unique_exprs[slot_with_exc]
            self.eval_error = excs.ExprEvalError(
                e, f'expression {e}', first_row.get_exc(e.slot_idx), exc_tb, input_vals, 0)
            self.exc_event.set()
            return

        for row in rows:
            assert row.has_exc(slot_with_exc)
            exc = row.get_exc(slot_with_exc)
            # propagate exception
            for slot_idx in np.nonzero(self.row_builder.transitive_dependents[slot_with_exc])[0].tolist():
                row.set_exc(slot_idx, exc)
        self.dispatch(rows)

    def dispatch(self, rows: list[exprs.DataRow]) -> None:
        if len(rows) == 0 or self.exc_event.is_set():
            return

        # slots ready for evaluation; rows x slots
        ready_slots = np.zeros((len(rows), self.row_builder.num_materialized), dtype=bool)
        for i, row in enumerate(rows):
            row.missing_slots &= row.has_val == False
            if row.missing_slots.sum() == 0:
                # all output slots have been materialized
                self.completed_rows.put_nowait(row)
            else:
                # dependencies of missing slots
                missing_dependencies = self.num_dependencies * row.missing_slots
                # determine ready slots that are not yet materialized and not yet scheduled
                num_mat_dependencies = np.sum(self.row_builder.dependencies * row.has_val, axis=1)
                num_missing = missing_dependencies - num_mat_dependencies
                ready_slots[i] = (num_missing == 0) & (row.is_scheduled == False) & row.missing_slots
                row.is_scheduled = row.is_scheduled | ready_slots[i]

            # clear intermediate values that are no longer needed (ie, all dependents are materialized)
            missing_dependents = np.sum(self.row_builder.dependencies[row.has_val == False], axis=0)
            gc_targets = (missing_dependents == 0) & (row.missing_dependents > 0) & self.gc_targets
            row.clear(gc_targets)
            row.missing_dependents = missing_dependents

        # schedule all ready slots
        for slot_idx in np.sum(ready_slots, axis=0).nonzero()[0]:
            ready_rows_v = ready_slots[:, slot_idx].flatten()
            _ = ready_rows_v.nonzero()
            ready_rows = [rows[i] for i in ready_rows_v.nonzero()[0]]
            _logger.debug(f'Scheduling {len(ready_rows)} rows for slot {slot_idx}')
            self.slot_evaluators[slot_idx].schedule(ready_rows, slot_idx)

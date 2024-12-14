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
import threading

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
            # TODO: is it possible for this to get cancelled after it got started?
            if asyncio.current_task().cancelled():
                return
            try:
                self.e.eval(row, self.dispatcher.row_builder)
            except Exception as exc:
                _, _, exc_tb = sys.exc_info()
                row.set_exc(self.e.slot_idx, exc)
                rows_with_excs.add(idx)
                self.dispatcher.dispatch_exc([row], self.e.slot_idx, exc_tb)
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

        for row in rows:
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


class AsyncExprEvalNode(ExecNode):
    """
    """
    num_dependencies: np.ndarray  # number of dependencies for our output slots; indexed by slot idx
    output_mask: np.ndarray  # of bool; True if this slot is part of our output
    slot_evaluators: dict[int, Evaluator]
    tasks: set[asyncio.Task]
    loop: Optional[asyncio.AbstractEventLoop]  # event loop for _main_loop()
    completed_rows: Optional[asyncio.Queue[exprs.DataRow]]  # rows ready to be returned
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
        input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.num_dependencies = np.sum(row_builder.dependencies, axis=1)
        self.output_mask = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.output_mask[output_slot_idxs] = True
        self.tasks = set()
        self.loop = None
        self.completed_rows = None

        self.gc_targets = np.ones(row_builder.num_materialized, dtype=bool)
        # we need to retain all slots that are part of the output
        self.gc_targets[[e.slot_idx for e in row_builder.output_exprs]] = False

        output_ctx = self.row_builder.create_eval_ctx(output_exprs, exclude=input_exprs)
        self.eval_ctx = np.zeros(row_builder.num_materialized, dtype=bool)
        self.eval_ctx[output_ctx.slot_idxs] = True
        self.exc_event = asyncio.Event()
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
            else:
                self.slot_evaluators[slot_idx] = DefaultExprEvaluator(expr, self)

    def __iter__(self) -> Iterator[DataRowBatch]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._iter_in_thread)
            return future.result()

    def _iter_in_thread(self) -> Iterator[DataRowBatch]:
        thread_id = threading.get_ident()
        _logger.debug(f'Starting thread {thread_id}')
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # create any Queue only *after* creating the event loop, otherwise it will be associated with the wrong loop
        self.completed_rows = asyncio.Queue[exprs.DataRow]()
        _logger.debug(f'Initialized thread {thread_id}')

        aiter = self._main_loop().__aiter__()
        try:
            while True:
                _logger.debug(f'Fetching batch: {thread_id}')
                batch: DataRowBatch = self.loop.run_until_complete(aiter.__anext__())
                _logger.debug(f'Yielding batch: {thread_id}')
                yield batch
        except StopAsyncIteration:
            assert len(self.tasks) == 0
        except excs.ExprEvalError as e:
            pending = asyncio.all_tasks(self.loop)
            assert len(pending) == 0
            raise e
        except Exception as e:
            print(f'Exception: {e}')
            pass
        finally:
            self.loop.close()

    async def _main_loop(self) -> AsyncIterator[DataRowBatch]:
        batch_rows: list[exprs.DataRow] = []
        input_row_iter = iter(self._input_rows())
        input_complete = False
        num_in_flight = 0  # number of dispatched rows that haven't been returned yet

        while True:
            if not input_complete and num_in_flight < self.MAX_IN_FLIGHT:
                # get more input rows scheduled
                _logger.debug(f'Trying to fetch {self.MAX_IN_FLIGHT - num_in_flight} input rows')
                input_rows = list(itertools.islice(input_row_iter, self.MAX_IN_FLIGHT - num_in_flight))
                _logger.debug(f'Fetched {len(input_rows)} input rows')
                self.num_input_rows += len(input_rows)
                input_complete = len(input_rows) == 0
                if input_complete:
                    _logger.debug('Input is complete')
                    for evaluator in self.slot_evaluators.values():
                        evaluator.close()
                num_in_flight += len(input_rows)
                self._init_input_rows(input_rows)
                self.dispatch(input_rows)

            # try to assemble output batch
            assert len(batch_rows) < self.BATCH_SIZE  # otherwise we would have yielded already
            _logger.debug(f'Assembling batch; #batch_rows={len(batch_rows)}')
            while len(batch_rows) < self.BATCH_SIZE:
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
                    done, pending = await asyncio.wait(
                        [self.completed_rows.get(), self.exc_event.wait()],
                        return_when=asyncio.FIRST_COMPLETED)
                    if self.exc_event.is_set():
                        # we got an exception that we need to propagate through __iter__();
                        # cancel all in-flight tasks first
                        for task in itertools.chain(pending, self.tasks):
                            task.cancel()
                        raise self.eval_error
                    row_ = done.pop().result()
                    assert isinstance(row_, exprs.DataRow)
                    row = row_
                batch_rows.append(row)
                num_in_flight -= 1

            _logger.debug(f'Finished assembly; #batch_rows={len(batch_rows)}')
            if len(batch_rows) == self.BATCH_SIZE or (input_complete and num_in_flight == 0):
                _logger.debug(f'Yielding batch of size {len(batch_rows)}')
                self.num_output_rows += len(batch_rows)
                yield DataRowBatch(tbl=None, row_builder=self.row_builder, rows=batch_rows)
                batch_rows = []

            if input_complete and num_in_flight == 0:
                assert len(batch_rows) == 0
                return

    def _init_input_rows(self, rows: list[exprs.DataRow]) -> None:
        """Set expr eval state in DataRow"""
        for row in rows:
            row.missing_dependents = np.sum(self.row_builder.dependencies[row.has_val == False], axis=0)
            row.missing_slots = self.eval_ctx & (row.has_val == False)

    def _input_rows(self) -> Iterator[exprs.DataRow]:
        for batch in self.input:
            _logger.debug(f'Fetched input batch of size {len(batch)}')
            yield from batch

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        if self.exc_event.is_set():
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
        if self.exc_event.is_set():
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

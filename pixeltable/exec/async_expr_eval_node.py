from __future__ import annotations

import abc
import asyncio
import datetime
import itertools
import logging
import sys
from dataclasses import dataclass
from types import TracebackType
from typing import Iterable, Iterator, Any, AsyncIterator, Protocol, Optional, Callable, Union, Awaitable

import numpy as np

import pixeltable.exceptions as excs
from pixeltable import env
from pixeltable import exprs
from pixeltable import func
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class Dispatcher(Protocol):
    """Row dispatcher used by Evaluators as a post-processing step after slot materialization"""
    tasks: set[asyncio.Task]
    row_builder: exprs.RowBuilder
    exc_event: asyncio.Event
    schedulers: dict[str, RateLimitsScheduler]

    def dispatch(self, rows: list[exprs.DataRow]) -> None:
        """Dispatches row slots to the appropriate schedulers; does not block"""
        ...

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        """Propagates exception in slot_with_exc to all dependent slots and dispatches the rest; does not block"""
        ...


class Evaluator(abc.ABC):
    """
    Base class for expression evaluators. Each DataRow slot is assigned an evaluator, which is responsible for the
    execution of the expression evaluation logic as well as the scheduling/task breakdown of that execution.

    Expected behavior:
    - all created tasks must be recorded in dispatcher.tasks
    - evaluators are responsible for aborting execution when a) the task is cancelled or b) when an exception occurred
      elsewhere (indicated by dispatcher.exc_event)
    """
    dispatcher: Dispatcher
    is_closed: bool

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.is_closed = False

    @abc.abstractmethod
    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        """
        Create tasks to evaluate the expression in the given slot for the given rows; must not block.

        """
        pass

    def _close(self) -> None:
        """Close the evaluator; must not block"""
        pass

    def close(self) -> None:
        """Indicates that there may not be any more rows getting scheduled"""
        self.is_closed = True
        self._close()


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


@dataclass
class FnCallArgs:
    fn_call: exprs.FunctionCall
    row: exprs.DataRow
    args: list[Any]
    kwargs: dict[str, Any]


class RateLimitsScheduler:
    """Scheduler for FunctionCalls with a RateLimitsInfo pool"""
    @dataclass
    class RequestInfo:
        submitted_at: datetime.datetime
        input_length: int
        max_tokens: int


    @dataclass
    class QueueItem:
        request: FnCallArgs
        num_retries: int

        def __lt__(self, other: RateLimitsScheduler.QueueItem) -> bool:
            return self.num_retries > other.num_retries


    resource_pool: str
    queue: asyncio.PriorityQueue[QueueItem]  # prioritizes retries
    loop_task: asyncio.Task
    dispatcher: Dispatcher

    # scheduling-related state
    pool_info: env.RateLimitsInfo
    acc_output_tokens: int  # accumulated output tokens since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    # token-related stats
    total_input_length: int
    total_input_tokens: int
    total_responses: int
    total_output_tokens: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        self.resource_pool = resource_pool
        self.queue = asyncio.PriorityQueue()
        self.dispatcher = dispatcher
        self.loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.tasks.add(self.loop_task)
        self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool)
        assert isinstance(self.pool_info, env.RateLimitsInfo)
        self.acc_output_tokens = 0
        self.num_in_flight = 0
        self.request_completed = asyncio.Event()
        self.total_input_length = 0
        self.total_input_tokens = 0
        self.total_responses = 0
        self.total_output_tokens = 0
        self.total_retried = 0

    def submit(self, item: FnCallArgs) -> None:
        self.queue.put_nowait(self.QueueItem(item, 0))

    def close(self) -> None:
        # TODO: do we need this?
        return
        #self.queue.put_nowait(None)

    async def _main_loop(self) -> None:
        item: Optional[RateLimitsScheduler.QueueItem] = None
        while True:
            if item is None:
                item = await self.queue.get()
                if item.num_retries > 0:
                    self.total_retried += 1

            now = datetime.datetime.now(tz=datetime.timezone.utc)
            if not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.num_retries, is_task=False)
                item = None
                continue

            # check rate limits
            rem_output_tokens = self._get_est_remaining_output_tokens()
            request_output_tokens = 1024  # TODO: get max_tokens arg from the call
            output_tokens_limit = self.pool_info.resource_limits['output_tokens'].limit
            # leave some headroom, we don't have perfect information
            if rem_output_tokens - request_output_tokens < 0.05 * output_tokens_limit:
            #if rem_output_tokens - request_output_tokens < 0.00 * output_tokens_limit:
                # wait for capacity to free up
                aws: list[Awaitable[None]] = []
                completed_aw: Optional[asyncio.Task] = None
                wait_for_reset: Optional[asyncio.Task] = None

                if self.num_in_flight > 0:
                    # a completed request can free up output_tokens capacity
                    self.request_completed.clear()
                    completed_aw = asyncio.create_task(self.request_completed.wait())
                    aws.append(completed_aw)
                    _logger.debug(f'waiting for completed request for {self.resource_pool}')

                reset_at = self.pool_info.resource_limits['output_tokens'].reset_at
                if reset_at > now:
                    # we're waiting for the rate limit to reset
                    wait_for_reset = asyncio.create_task(asyncio.sleep((reset_at - now).total_seconds()))
                    aws.append(wait_for_reset)
                    _logger.debug(f'waiting for rate limit reset for {self.resource_pool}')

                if len(aws) == 0:
                    # we have nothing in particular to wait for: wait for an arbitrary amount of time and then
                    # re-evaluate the rate limits
                    aws.append(asyncio.sleep(1.0))
                    _logger.debug(f'waiting for 1.0 for {self.resource_pool}')

                done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if completed_aw in done:
                    _logger.debug(f'wait(): completed request for {self.resource_pool}')
                if wait_for_reset in done:
                    _logger.debug(f'wait(): rate limit reset for {self.resource_pool}')
                    # force re-acquisition of rate limits before making any scheduling decisions
                    self.pool_info.reset()

                # re-evaluate current capacity for current item
                continue

            assert rem_output_tokens >= request_output_tokens
            self.acc_output_tokens += request_output_tokens
            _logger.debug(f'creating task for {self.resource_pool}, rem_output_tokens={rem_output_tokens}')
            self.num_in_flight += 1
            task = asyncio.create_task(self._exec(item.request, item.num_retries, is_task=True))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.tasks.discard)
            item = None

    def _get_est_remaining_output_tokens(self) -> int:
        reported = self.pool_info.resource_limits['output_tokens'].remaining
        recorded_at = self.pool_info.resource_limits['output_tokens'].recorded_at
        _logger.debug(f'est remaining output_tokens for {self.resource_pool}: report={reported} adjustment={-self.acc_output_tokens} recorded_at={recorded_at.strftime(self.TIME_FORMAT)}')
        return max(0, reported - self.acc_output_tokens)

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert not request.row.has_val[request.fn_call.slot_idx] and not request.row.has_exc(request.fn_call.slot_idx)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}')
            result = await request.fn_call.fn.aexec(*request.args, **request.kwargs)
            request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}')

            # update token-related stats
            self.total_input_length += 10  # TODO: get the actual input length
            self.total_input_tokens += result['usage']['input_tokens']
            self.total_responses += 1
            self.total_output_tokens += result['usage']['output_tokens']

            # purge accumulated resource usage, now that we have a new report
            self.acc_output_tokens = 0

            self.dispatcher.dispatch([request.row])
        except Exception as exc:
            import anthropic
            if isinstance(exc, anthropic.APIStatusError):
                # retry this request, if it's retryable
                _logger.debug(f'output_tokens RateLimitError: total_output_tokens={self.total_output_tokens}')
                _logger.debug(f'headers={exc.response.headers}')
                should_retry_str = exc.response.headers.get('x-should-retry', '')
                if should_retry_str.lower() == 'true' and num_retries < self.MAX_RETRIES:
                    retry_after_str = exc.response.headers.get('retry-after', '1')
                    retry_after_secs = int(retry_after_str)
                    await asyncio.sleep(retry_after_secs)
                    self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                    return

            # record the exception
            _, _, exc_tb = sys.exc_info()
            request.row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc([request.row], request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(
                f'Scheduler stats: total_input_length={self.total_input_length}, total_input_tokens={self.total_input_tokens}, total_responses={self.total_responses}, total_output_tokens={self.total_output_tokens} total_retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


class OpenAIRateLimitsScheduler:
    """Scheduler for FunctionCalls with a RateLimitsInfo pool"""
    @dataclass
    class RequestInfo:
        submitted_at: datetime.datetime
        input_length: int
        max_tokens: int


    @dataclass
    class QueueItem:
        request: FnCallArgs
        num_retries: int

        def __lt__(self, other: RateLimitsScheduler.QueueItem) -> bool:
            return self.num_retries > other.num_retries


    resource_pool: str
    queue: asyncio.PriorityQueue[QueueItem]  # prioritizes retries
    loop_task: asyncio.Task
    dispatcher: Dispatcher

    # scheduling-related state
    pool_info: env.RateLimitsInfo
    acc_output_tokens: int  # accumulated output tokens since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    # token-related stats
    total_input_length: int
    total_input_tokens: int
    total_responses: int
    total_output_tokens: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        self.resource_pool = resource_pool
        self.queue = asyncio.PriorityQueue()
        self.dispatcher = dispatcher
        self.loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.tasks.add(self.loop_task)
        self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool)
        assert isinstance(self.pool_info, env.RateLimitsInfo)
        self.acc_output_tokens = 0
        self.num_in_flight = 0
        self.request_completed = asyncio.Event()
        self.total_input_length = 0
        self.total_input_tokens = 0
        self.total_responses = 0
        self.total_output_tokens = 0
        self.total_retried = 0

    def submit(self, item: FnCallArgs) -> None:
        self.queue.put_nowait(self.QueueItem(item, 0))

    def close(self) -> None:
        # TODO: do we need this?
        return
        #self.queue.put_nowait(None)

    async def _main_loop(self) -> None:
        item: Optional[RateLimitsScheduler.QueueItem] = None
        while True:
            if item is None:
                item = await self.queue.get()
                if item.num_retries > 0:
                    self.total_retried += 1

            now = datetime.datetime.now(tz=datetime.timezone.utc)
            if not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.num_retries, is_task=False)
                item = None
                continue

            # check rate limits
            rem_output_tokens = self._get_est_remaining_output_tokens()
            request_output_tokens = 1024  # TODO: get max_tokens arg from the call
            output_tokens_limit = self.pool_info.resource_limits['output_tokens'].limit
            # leave some headroom, we don't have perfect information
            if rem_output_tokens - request_output_tokens < 0.05 * output_tokens_limit:
                #if rem_output_tokens - request_output_tokens < 0.00 * output_tokens_limit:
                # wait for capacity to free up
                aws: list[Awaitable[None]] = []
                completed_aw: Optional[asyncio.Task] = None
                wait_for_reset: Optional[asyncio.Task] = None

                if self.num_in_flight > 0:
                    # a completed request can free up output_tokens capacity
                    self.request_completed.clear()
                    completed_aw = asyncio.create_task(self.request_completed.wait())
                    aws.append(completed_aw)
                    _logger.debug(f'waiting for completed request for {self.resource_pool}')

                reset_at = self.pool_info.resource_limits['output_tokens'].reset_at
                if reset_at > now:
                    # we're waiting for the rate limit to reset
                    wait_for_reset = asyncio.create_task(asyncio.sleep((reset_at - now).total_seconds()))
                    aws.append(wait_for_reset)
                    _logger.debug(f'waiting for rate limit reset for {self.resource_pool}')

                if len(aws) == 0:
                    # we have nothing in particular to wait for: wait for an arbitrary amount of time and then
                    # re-evaluate the rate limits
                    aws.append(asyncio.sleep(1.0))
                    _logger.debug(f'waiting for 1.0 for {self.resource_pool}')

                done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if completed_aw in done:
                    _logger.debug(f'wait(): completed request for {self.resource_pool}')
                if wait_for_reset in done:
                    _logger.debug(f'wait(): rate limit reset for {self.resource_pool}')
                    # force re-acquisition of rate limits before making any scheduling decisions
                    self.pool_info.reset()

                # re-evaluate current capacity for current item
                continue

            assert rem_output_tokens >= request_output_tokens
            self.acc_output_tokens += request_output_tokens
            _logger.debug(f'creating task for {self.resource_pool}, rem_output_tokens={rem_output_tokens}')
            self.num_in_flight += 1
            task = asyncio.create_task(self._exec(item.request, item.num_retries, is_task=True))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.tasks.discard)
            item = None

    def _get_est_remaining_output_tokens(self) -> int:
        reported = self.pool_info.resource_limits['output_tokens'].remaining
        recorded_at = self.pool_info.resource_limits['output_tokens'].recorded_at
        _logger.debug(f'est remaining output_tokens for {self.resource_pool}: report={reported} adjustment={-self.acc_output_tokens} recorded_at={recorded_at.strftime(self.TIME_FORMAT)}')
        return max(0, reported - self.acc_output_tokens)

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert not request.row.has_val[request.fn_call.slot_idx] and not request.row.has_exc(request.fn_call.slot_idx)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}')
            result = await request.fn_call.fn.aexec(*request.args, **request.kwargs)
            request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}')

            # update token-related stats
            self.total_input_length += 10  # TODO: get the actual input length
            self.total_input_tokens += result['usage']['input_tokens']
            self.total_responses += 1
            self.total_output_tokens += result['usage']['output_tokens']

            # purge accumulated resource usage, now that we have a new report
            self.acc_output_tokens = 0

            self.dispatcher.dispatch([request.row])
        except Exception as exc:
            import anthropic
            if isinstance(exc, anthropic.APIStatusError):
                # retry this request, if it's retryable
                _logger.debug(f'output_tokens RateLimitError: total_output_tokens={self.total_output_tokens}')
                _logger.debug(f'headers={exc.response.headers}')
                should_retry_str = exc.response.headers.get('x-should-retry', '')
                if should_retry_str.lower() == 'true' and num_retries < self.MAX_RETRIES:
                    retry_after_str = exc.response.headers.get('retry-after', '1')
                    retry_after_secs = int(retry_after_str)
                    await asyncio.sleep(retry_after_secs)
                    self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                    return

            # record the exception
            _, _, exc_tb = sys.exc_info()
            request.row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc([request.row], request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(
                f'Scheduler stats: total_input_length={self.total_input_length}, total_input_tokens={self.total_input_tokens}, total_responses={self.total_responses}, total_output_tokens={self.total_output_tokens} total_retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


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
                result_batch = self.fn.exec_batch(*arg_batches, **kwarg_batches)
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


class CircularRowBuffer:
    """Fixed-length circular buffer of DataRows; knows how to maintain input order"""

    size: int
    maintain_order: bool  # True if we're returning rows order by position
    num_rows: int  # number of rows in the buffer
    num_ready: int  # number of consecutive non-None rows at head
    buffer: np.ndarray  # of object
    head_idx: int  # index of beginning of the buffer
    head_pos: int  # row position of the beginning of the buffer

    def __init__(self, size: int, maintain_order: bool):
        self.size = size
        self.maintain_order = maintain_order
        self.num_rows = 0
        self.num_ready = 0
        self.buffer = np.full(size, None, dtype=object)
        self.head_pos = 0
        self.head_idx = 0

    def add_row(self, row: exprs.DataRow, pos: Optional[int]) -> None:
        assert pos is None or (pos - self.head_pos < self.size), f'{pos} {self.head_pos} {self.size}'
        offset: int  # of new row from head
        if self.maintain_order:
            offset = pos - self.head_pos
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
        self.head_pos += n
        self.head_idx = (self.head_idx + n) % self.size
        self.num_rows -= n
        self.num_ready -= n
        return rows


class AsyncExprEvalNode(ExecNode):
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
    - local model inference on gpu: currently, no attempt is made to ensure that models can fit onto the gpu
      simultaneously, which will cause errors; instead, the execution should be divided into sequential phases, each
      of which only contains a subset of the models which is known to fit onto the gpu simultaneously
    """
    maintain_input_order: bool  # True if we're returning rows in the order we received them from our input
    num_dependencies: np.ndarray  # number of dependencies for our output slots; indexed by slot idx
    outputs: np.ndarray  # bool per slot; True if this slot is part of our output
    slot_evaluators: dict[int, Evaluator]  # key: slot idx
    schedulers: dict[str, RateLimitsScheduler]  # key: resource pool name
    gc_targets: np.ndarray  # bool per slot; True if this is an intermediate expr (ie, not part of our output)
    eval_ctx: np.ndarray  # bool per slot; EvalCtx.slot_idxs as a mask

    # execution state
    tasks: set[asyncio.Task]  # collects all running tasks to prevent them from getting gc'd
    exc_event: asyncio.Event  # set if an exception needs to be propagated
    error: Optional[Union[excs.Error, excs.ExprEvalError]]  # exception that needs to be propagated
    completed_rows: asyncio.Queue[exprs.DataRow]  # rows that have completed evaluation
    completed_event: asyncio.Event  # set when completed_rows is non-empty
    input_iter: AsyncIterator[DataRowBatch]
    current_input_batch: Optional[DataRowBatch]  # batch from which we're currently consuming rows
    input_row_idx: int  # next row to consume from current_input_batch
    next_input_batch: Optional[DataRowBatch]  # read-ahead input batch
    avail_input_rows: int  # total number across both current_/next_input_batch
    input_complete: bool  # True if we've received all input batches
    num_in_flight: int  # number of dispatched rows that haven't completed
    row_pos_map: Optional[dict[int, int]]  # id(row) -> position of row in input; only set if maintain_input_order
    output_buffer: CircularRowBuffer  # holds rows that are ready to be returned, in order

    # debugging
    num_input_rows: int
    num_output_rows: int

    BATCH_SIZE = 12
    MAX_BUFFERED_ROWS = 512  # maximum number of rows that have been dispatched but not yet returned

    def __init__(
        self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr], input_exprs: Iterable[exprs.Expr],
        maintain_input_order: bool, input: ExecNode
    ):
        super().__init__(row_builder, output_exprs, input_exprs, input)
        self.maintain_input_order = maintain_input_order
        self.num_dependencies = np.sum(row_builder.dependencies, axis=1)
        self.outputs = np.zeros(row_builder.num_materialized, dtype=bool)
        output_slot_idxs = [e.slot_idx for e in output_exprs]
        self.outputs[output_slot_idxs] = True
        self.tasks = set()

        self.gc_targets = np.ones(row_builder.num_materialized, dtype=bool)
        # we need to retain all slots that are part of the output
        self.gc_targets[[e.slot_idx for e in row_builder.output_exprs]] = False

        output_ctx = self.row_builder.create_eval_ctx(output_exprs, exclude=input_exprs)
        self.eval_ctx = np.zeros(row_builder.num_materialized, dtype=bool)
        self.eval_ctx[output_ctx.slot_idxs] = True
        self.error = None

        self.input_iter = self.input.__aiter__()
        self.current_input_batch = None
        self.next_input_batch = None
        self.input_row_idx = 0
        self.avail_input_rows = 0
        self.input_complete = False
        self.num_in_flight = 0
        self.row_pos_map = {}
        self.output_buffer = CircularRowBuffer(self.MAX_BUFFERED_ROWS, maintain_order=self.maintain_input_order)

        self.num_input_rows = 0
        self.num_output_rows = 0

        self.slot_evaluators = {}
        self.schedulers = {}
        self._init_slot_evaluators()

    def _init_slot_evaluators(self) -> None:
        """Create slot evaluators and resource pool schedulers"""
        resource_pools: set[str] = set()
        for slot_idx in range(self.row_builder.num_materialized):
            expr = self.row_builder.unique_exprs[slot_idx]
            if (
                isinstance(expr, exprs.FunctionCall)
                # ExprTemplateFunction and AggregateFunction calls are best handled by FunctionCall.eval()
                and not isinstance(expr.fn, func.ExprTemplateFunction)
                and not isinstance(expr.fn, func.AggregateFunction)
            ):
                if expr.resource_pool is not None:
                    resource_pools.add(expr.resource_pool)
                self.slot_evaluators[slot_idx] = FnCallEvaluator(expr, self)
            else:
                self.slot_evaluators[slot_idx] = DefaultExprEvaluator(expr, self)

    async def _fetch_input_batch(self) -> None:
        """
        Fetches another batch from our input or sets input_complete to True if there are no more batches.

        - stores the batch in current_input_batch, if not already set, or next_input_batch
        - updates row_pos_map, if needed
        """
        assert not self.input_complete
        try:
            batch = await self.input_iter.__anext__()
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
        except StopAsyncIteration:
            self.input_complete = True
            for evaluator in self.slot_evaluators.values():
                evaluator.close()
        except excs.Error as err:
            self.error = err
            self.exc_event.set()
        # TODO: should we also handle Exception here and create an excs.Error from it?

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
            rows = self.current_input_batch.rows[self.input_row_idx:self.input_row_idx + num_rows]
            self.input_row_idx += num_rows
        else:
            # we need rows from both current_/next_input_batch
            rows = self.current_input_batch.rows[self.input_row_idx:]
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

        self._init_input_rows(rows)
        self.dispatch(rows)

    def _log_state(self, prefix: str) -> None:
        _logger.debug(
            f'{prefix}: #in-flight={self.num_in_flight} #complete={self.completed_rows.qsize()} '
            f'#output-buffer={self.output_buffer.num_rows} #ready={self.output_buffer.num_ready} '
            f'total-buffered={self.total_buffered} #avail={self.avail_input_rows}'
        )

    def _init_schedulers(self) -> None:
        resource_pools = {
            eval.fn_call.resource_pool for eval in self.slot_evaluators.values() if isinstance(eval, FnCallEvaluator)
        }
        resource_pools = {pool for pool in resource_pools if pool is not None}
        for pool_name in resource_pools:
            self.schedulers[pool_name] = RateLimitsScheduler(pool_name, self)

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
        row: exprs.DataRow
        exc_event_aw = asyncio.create_task(self.exc_event.wait(), name='exc_event.wait()')
        input_batch_aw: Optional[asyncio.Task] = None
        completed_aw: Optional[asyncio.Task] = None

        try:
            while True:
                # process completed rows before doing anything else
                while not self.completed_rows.empty():
                    #self._log_state('processing completed')
                    # move completed rows to output buffer
                    while not self.completed_rows.empty():
                        row = self.completed_rows.get_nowait()
                        self.output_buffer.add_row(row, self.row_pos_map.pop(id(row), None))

                    self._log_state('processed completed')
                    # return as many batches as we have available
                    while self.output_buffer.num_ready >= self.BATCH_SIZE:
                        batch_rows = self.output_buffer.get_rows(self.BATCH_SIZE)
                        self.num_output_rows += len(batch_rows)
                        # make sure we top up our in-flight rows before yielding
                        self._dispatch_input_rows()
                        self._log_state(f'yielding {len(batch_rows)} rows')
                        yield DataRowBatch(tbl=None, row_builder=self.row_builder, rows=batch_rows)
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
                        yield DataRowBatch(tbl=None, row_builder=self.row_builder, rows=batch_rows)

                    assert self.output_buffer.num_rows == 0
                    return

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
                done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)

                if self.exc_event.is_set():
                    # we got an exception that we need to propagate through __iter__()
                    _logger.debug(f'Propagating exception {self.error}')
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

    def _init_input_rows(self, rows: list[exprs.DataRow]) -> None:
        """Set execution state in DataRow"""
        for row in rows:
            row.missing_dependents = np.sum(self.row_builder.dependencies[row.has_val == False], axis=0)
            row.missing_slots = self.eval_ctx & (row.has_val == False)

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        """Propagate exception to main event loop or to dependent slots, depending on ignore_errors"""
        if len(rows) == 0 or self.exc_event.is_set():
            return

        if not self.ctx.ignore_errors:
            dependency_idxs = [e.slot_idx for e in self.row_builder.unique_exprs[slot_with_exc].dependencies()]
            first_row = rows[0]
            input_vals = [first_row[idx] for idx in dependency_idxs]
            e = self.row_builder.unique_exprs[slot_with_exc]
            self.error = excs.ExprEvalError(
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
        """Dispatch rows to slot evaluators, based on materialized dependencies"""
        if len(rows) == 0 or self.exc_event.is_set():
            return

        # slots ready for evaluation; rows x slots
        ready_slots = np.zeros((len(rows), self.row_builder.num_materialized), dtype=bool)
        completed_rows = np.zeros(len(rows), dtype=bool)
        for i, row in enumerate(rows):
            row.missing_slots &= row.has_val == False
            if row.missing_slots.sum() == 0:
                # all output slots have been materialized
                completed_rows[i] = True
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

        if np.any(completed_rows):
            completed_idxs = list(completed_rows.nonzero()[0])
            for i in completed_idxs:
                self.completed_rows.put_nowait(rows[i])
            self.completed_event.set()
            self.num_in_flight -= len(completed_idxs)

        # schedule all ready slots
        for slot_idx in np.sum(ready_slots, axis=0).nonzero()[0]:
            ready_rows_v = ready_slots[:, slot_idx].flatten()
            _ = ready_rows_v.nonzero()
            ready_rows = [rows[i] for i in ready_rows_v.nonzero()[0]]
            _logger.debug(f'Scheduling {len(ready_rows)} rows for slot {slot_idx}')
            self.slot_evaluators[slot_idx].schedule(ready_rows, slot_idx)

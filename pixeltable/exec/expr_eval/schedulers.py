from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import sys
from dataclasses import dataclass
from typing import Optional, Awaitable, Collection

from pixeltable import env
from pixeltable import func
from .globals import Scheduler, FnCallArgs, Dispatcher

_logger = logging.getLogger('pixeltable')


class RateLimitsScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a RateLimitsInfo pool

    TODO:
    - limit the number of in-flight requests based on the open file limit
    """
    @dataclass(frozen=True)
    class QueueItem:
        request: FnCallArgs
        num_retries: int

        def __lt__(self, other: RateLimitsScheduler.QueueItem) -> bool:
            # prioritize by number of retries
            return self.num_retries > other.num_retries

    resource_pool: str
    queue: asyncio.PriorityQueue[QueueItem]  # prioritizes retries
    loop_task: asyncio.Task
    dispatcher: Dispatcher
    get_request_resources_param_names: list[str]  # names of parameters of RateLimitsInfo.get_request_resources()

    # scheduling-related state
    pool_info: Optional[env.RateLimitsInfo]
    est_usage: dict[str, int]  # value per resource; accumulated estimates since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    total_requests: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        self.resource_pool = resource_pool
        self.queue = asyncio.PriorityQueue()
        self.dispatcher = dispatcher
        self.loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.tasks.add(self.loop_task)
        self.loop_task.add_done_callback(self.dispatcher.done_cb)
        self.pool_info = None  # initialized in _main_loop by the first request
        self.est_usage = {}
        self.num_in_flight = 0
        self.request_completed = asyncio.Event()
        self.total_requests = 0
        self.total_retried = 0
        self.get_request_resources_param_names = []

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('rate-limits:')

    def submit(self, item: FnCallArgs) -> None:
        self.queue.put_nowait(self.QueueItem(item, 0))

    def close(self) -> None:
        # TODO: do we need this?
        return

    async def _main_loop(self) -> None:
        item: Optional[RateLimitsScheduler.QueueItem] = None
        while True:
            if item is None:
                item = await self.queue.get()
                if item.num_retries > 0:
                    self.total_retried += 1

            now = datetime.datetime.now(tz=datetime.timezone.utc)
            if self.pool_info is None or not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.num_retries, is_task=False)
                item = None
                # if this was the first request, it created the pool_info
                if self.pool_info is None:
                    self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool, None)
                    if self.pool_info is None:
                        # we still don't have rate limits, wait for the next request
                        continue
                    assert isinstance(self.pool_info, env.RateLimitsInfo)
                    assert hasattr(self.pool_info, 'get_request_resources')
                    sig = inspect.signature(self.pool_info.get_request_resources)  # type: ignore
                    self.get_request_resources_param_names = [p.name for p in sig.parameters.values()]
                    self.est_usage = {r: 0 for r in self._resources}
                continue

            # check rate limits
            request_resources = self._get_request_resources(item.request)
            limits_info = self._check_resource_limits(request_resources)
            if limits_info is not None:
                # limits_info's resource is depleted, wait for capacity to free up
                aws: list[Awaitable[None]] = []
                completed_aw: Optional[asyncio.Task] = None
                wait_for_reset: Optional[asyncio.Task] = None

                if self.num_in_flight > 0:
                    # a completed request can free up capacity
                    self.request_completed.clear()
                    completed_aw = asyncio.create_task(self.request_completed.wait())
                    aws.append(completed_aw)
                    _logger.debug(f'waiting for completed request for {self.resource_pool}')

                reset_at = limits_info.reset_at
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
                    # force waiting for another rate limit report before making any scheduling decisions
                    self.pool_info.reset()

                # re-evaluate current capacity for current item
                continue

            # we have a new in-flight request
            for resource, val in request_resources.items():
                self.est_usage[resource] += val
            _logger.debug(f'creating task for {self.resource_pool}')
            self.num_in_flight += 1
            task = asyncio.create_task(self._exec(item.request, item.num_retries, is_task=True))
            self.dispatcher.tasks.add(task)
            task.add_done_callback(self.dispatcher.done_cb)
            item = None

    @property
    def _resources(self) -> Collection[str]:
        return self.pool_info.resource_limits.keys() if self.pool_info is not None else []

    def _get_request_resources(self, request: FnCallArgs) -> dict[str, int]:
        kwargs_batch = request.fn_call.get_param_values(self.get_request_resources_param_names, request.rows)
        if not request.is_batched:
            return self.pool_info.get_request_resources(**kwargs_batch[0])
        else:
            batch_kwargs = {k: [d[k] for d in kwargs_batch] for k in kwargs_batch[0]}
            constant_kwargs, batch_kwargs = request.pxt_fn.create_batch_kwargs(batch_kwargs)
            return self.pool_info.get_request_resources(**constant_kwargs, **batch_kwargs)


    def _check_resource_limits(self, request_resources: dict[str, int]) -> Optional[env.RateLimitInfo]:
        """Returns the most exhausted resource, relative to its limit, or None if all resources are within limits"""
        candidates: list[tuple[env.RateLimitInfo, float]] = []  # (info, relative usage)
        for resource, usage in request_resources.items():
            # 0.05: leave some headroom, we don't have perfect information
            info = self.pool_info.resource_limits[resource]
            est_remaining = info.remaining - self.est_usage[resource] - usage
            if est_remaining < 0.05 * info.limit:
                candidates.append((info, est_remaining / info.limit))
        if len(candidates) == 0:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert all(not row.has_val[request.fn_call.slot_idx] for row in request.rows)
        assert all(not row.has_exc(request.fn_call.slot_idx) for row in request.rows)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            pxt_fn = request.fn_call.fn
            assert isinstance(pxt_fn, func.CallableFunction)
            _logger.debug(f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}, batch_size={len(request.rows)}')
            self.total_requests += 1
            if request.is_batched:
                batch_result = await pxt_fn.aexec_batch(*request.batch_args, **request.batch_kwargs)
                assert len(batch_result) == len(request.rows)
                for row, result in zip(request.rows, batch_result):
                    row[request.fn_call.slot_idx] = result
            else:
                result = await pxt_fn.aexec(*request.args, **request.kwargs)
                request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}, batch_size={len(request.rows)}')

            # purge accumulated usage estimate, now that we have a new report
            self.est_usage = {r: 0 for r in self._resources}

            self.dispatcher.dispatch(request.rows)
        except Exception as exc:
            if num_retries < self.MAX_RETRIES and self.pool_info is not None:
                retry_delay = self.pool_info.get_retry_delay(exc)
                if retry_delay is not None:
                    self.total_retried += 1
                    await asyncio.sleep(retry_delay)
                    self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                    return
            # TODO: update resource limits reported in exc.response.headers, if present

            # record the exception
            _, _, exc_tb = sys.exc_info()
            for row in request.rows:
                row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(request.rows, request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(
                f'Scheduler stats: #requests={self.total_requests}, #retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


# all concrete Scheduler subclasses that implement matches()
SCHEDULERS = [RateLimitsScheduler]

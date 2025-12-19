from __future__ import annotations

import asyncio
import datetime
import inspect
import logging
import math
import sys
import time
from typing import Awaitable, Collection

from pixeltable import env, func
from pixeltable.config import Config
from pixeltable.utils.http import exponential_backoff, is_retriable_error

from .globals import Dispatcher, ExprEvalCtx, FnCallArgs, Scheduler

_logger = logging.getLogger('pixeltable')

__all__ = ['RateLimitsScheduler', 'RequestRateScheduler']


class RateLimitsScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a RateLimitsInfo pool, which provides information about actual resource usage.

    Scheduling strategy:
    - try to stay below resource limits by utilizing reported RateLimitInfo.remaining
    - also take into account the estimated resource usage for in-flight requests
      (obtained via RateLimitsInfo.get_request_resources())
    - issue synchronous requests when we don't have a RateLimitsInfo yet or when we depleted a resource and need to
      wait for a reset

    TODO:
    - limit the number of in-flight requests based on the open file limit
    """

    get_request_resources_param_names: list[str]  # names of parameters of RateLimitsInfo.get_request_resources()

    # scheduling-related state
    pool_info: env.RateLimitsInfo | None
    est_usage: dict[str, int]  # value per resource; accumulated estimates since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    total_requests: int
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(resource_pool, dispatcher)
        loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.register_task(loop_task)
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

    def _set_pool_info(self) -> None:
        """Initialize pool_info with the RateLimitsInfo for the resource pool, if available"""
        if self.pool_info is not None:
            return
        self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool, None)
        if self.pool_info is None:
            return
        assert isinstance(self.pool_info, env.RateLimitsInfo)
        assert hasattr(self.pool_info, 'get_request_resources')
        sig = inspect.signature(self.pool_info.get_request_resources)
        self.get_request_resources_param_names = [p.name for p in sig.parameters.values()]
        self.est_usage = dict.fromkeys(self._resources, 0)

    async def _main_loop(self) -> None:
        item: RateLimitsScheduler.QueueItem | None = None
        while True:
            if item is None:
                item = await self.queue.get()
                assert isinstance(item.request.fn_call.fn, func.CallableFunction)
                assert '_runtime_ctx' in item.request.fn_call.fn.signature.system_parameters
                if item.num_retries > 0:
                    self.total_retried += 1

            if self.pool_info is None or not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.exec_ctx, item.num_retries, is_task=False)
                _logger.debug(f'initialized rate limits for {self.resource_pool}')
                item = None
                # if this was the first request, it created the pool_info
                if self.pool_info is None:
                    self._set_pool_info()
                continue

            # check rate limits
            request_resources = self._get_request_resources(item.request)
            resource_delay = self._resource_delay(request_resources)
            aws: list[Awaitable[None]] = []
            completed_aw: asyncio.Task | None = None
            wait_for_reset: asyncio.Task | None = None
            if resource_delay > 0:
                # Some resource or resources are nearing depletion

                if self.num_in_flight > 0:
                    # a completed request can free up capacity
                    self.request_completed.clear()
                    completed_aw = asyncio.create_task(self.request_completed.wait())
                    aws.append(completed_aw)
                    _logger.debug(f'waiting for completed request for {self.resource_pool}')

                # Schedule a sleep until sufficient resources are available
                wait_for_reset = asyncio.create_task(asyncio.sleep(resource_delay))
                aws.append(wait_for_reset)
                _logger.debug(f'waiting {resource_delay:.1f}s for resource availability')

            if len(aws) > 0:
                # we have something to wait for
                done, pending = await asyncio.wait(aws, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                if completed_aw in done:
                    _logger.debug(f'wait(): completed request for {self.resource_pool}')
                # re-evaluate current capacity for current item
                continue

            # we have a new in-flight request
            for resource, val in request_resources.items():
                self.est_usage[resource] = self.est_usage.get(resource, 0) + val
            _logger.debug(f'creating task for {self.resource_pool}')
            self.num_in_flight += 1
            task = asyncio.create_task(self._exec(item.request, item.exec_ctx, item.num_retries, is_task=True))
            self.dispatcher.register_task(task)
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

    def _resource_delay(self, request_resources: dict[str, int]) -> float:
        """For the provided resources and usage, attempts to estimate the time to wait until sufficient resources are
        available."""
        highest_wait = 0.0
        highest_wait_resource = None
        for resource, usage in request_resources.items():
            info = self.pool_info.resource_limits[resource]
            # Note: usage and est_usage are estimated costs of requests, and it may be way off (for example, if max
            # tokens is unspecified for an openAI request).
            time_until = info.estimated_resource_refill_delay(
                math.ceil(info.limit * env.TARGET_RATE_LIMIT_RESOURCE_FRACT + usage + self.est_usage.get(resource, 0))
            )
            if time_until is not None and highest_wait < time_until:
                highest_wait = time_until
                highest_wait_resource = resource
        _logger.debug(f'Determined wait time of {highest_wait:.1f}s for resource {highest_wait_resource}')
        return highest_wait

    async def _exec(self, request: FnCallArgs, exec_ctx: ExprEvalCtx, num_retries: int, is_task: bool) -> None:
        assert all(not row.has_val[request.fn_call.slot_idx] for row in request.rows)
        assert all(not row.has_exc(request.fn_call.slot_idx) for row in request.rows)

        start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
        try:
            pxt_fn = request.fn_call.fn
            assert isinstance(pxt_fn, func.CallableFunction)
            _logger.debug(
                f'scheduler {self.resource_pool}: '
                f'start evaluating slot {request.fn_call.slot_idx}, batch_size={len(request.rows)}'
            )
            self.total_requests += 1
            if request.is_batched:
                batch_result = await pxt_fn.aexec_batch(*request.batch_args, **request.batch_kwargs)
                assert len(batch_result) == len(request.rows)
                for row, result in zip(request.rows, batch_result):
                    row[request.fn_call.slot_idx] = result
            else:
                request_kwargs = {**request.kwargs, '_runtime_ctx': env.RuntimeCtx(is_retry=num_retries > 0)}
                result = await pxt_fn.aexec(*request.args, **request_kwargs)
                request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(
                f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} '
                f'in {end_ts - start_ts}, batch_size={len(request.rows)}'
            )

            # purge accumulated usage estimate, now that we have a new report
            self.est_usage = dict.fromkeys(self._resources, 0)

            self.dispatcher.dispatch(request.rows, exec_ctx)
        except Exception as exc:
            _logger.exception(f'scheduler {self.resource_pool}: exception in slot {request.fn_call.slot_idx}: {exc}')
            if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
                _logger.debug(f'scheduler {self.resource_pool}: exception headers: {exc.response.headers}')

            # If pool info is available, attempt to retry based on the resource information
            # Pool info may not be available yet if the exception occurred before the UDF set it
            if self.pool_info is not None:
                self.pool_info.record_exc(start_ts, exc)

                if num_retries < self.MAX_RETRIES:
                    retry_delay = self.pool_info.get_retry_delay(exc, num_retries)
                    if retry_delay is None:
                        # The resource pool did not recognize it as a retriable error. Try our generic best-effort logic
                        # before giving up.
                        is_retriable, retry_delay = is_retriable_error(exc)
                        if is_retriable:
                            retry_delay = retry_delay or exponential_backoff(num_retries)
                    if retry_delay is not None:
                        self.total_retried += 1
                        _logger.debug(
                            f'scheduler {self.resource_pool}: sleeping {retry_delay:.2f}s before retrying'
                            f' attempt {num_retries} based on the information in the error'
                        )
                        await asyncio.sleep(retry_delay)
                        self.queue.put_nowait(self.QueueItem(request, num_retries + 1, exec_ctx))
                        return

            # record the exception
            _, _, exc_tb = sys.exc_info()
            for row in request.rows:
                row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(request.rows, request.fn_call.slot_idx, exc_tb, exec_ctx)
        finally:
            _logger.debug(f'Scheduler stats: #requests={self.total_requests}, #retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


class RequestRateScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a fixed request rate limit and no runtime resource usage reports.

    Rate limits are supplied in the config, in one of two ways:
    - resource_pool='request-rate:<endpoint>':
      * a single rate limit for all calls against that endpoint
      * in the config: section '<endpoint>', key 'rate_limit'
    - resource_pool='request-rate:<endpoint>:<model>':
        * a single rate limit for all calls against that model
        * in the config: section '<endpoint>.rate_limits', key '<model>'
    - if no rate limit is found in the config, uses a default of 600 RPM

    TODO:
    - adaptive rate limiting based on 429 errors
    """

    secs_per_request: float  # inverted rate limit
    num_in_flight: int
    total_requests: int
    total_retried: int
    total_errors: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 3
    DEFAULT_RATE_LIMIT = 600  # requests per minute

    # Exponential backoff defaults
    BASE_RETRY_DELAY = 1.0  # in seconds
    MAX_RETRY_DELAY = 60.0  # in seconds

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(resource_pool, dispatcher)
        loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.register_task(loop_task)
        self.num_in_flight = 0
        self.total_requests = 0
        self.total_retried = 0
        self.total_errors = 0

        # try to get the rate limit from the config
        elems = resource_pool.split(':')
        section: str
        key: str
        if len(elems) == 2:
            # resource_pool: request-rate:endpoint
            _, endpoint = elems
            section = endpoint
            key = 'rate_limit'
        else:
            # resource_pool: request-rate:endpoint:model
            assert len(elems) == 3
            _, endpoint, model = elems
            section = f'{endpoint}.rate_limits'
            key = model
        requests_per_min = Config.get().get_int_value(key, section=section)
        requests_per_min = requests_per_min or self.DEFAULT_RATE_LIMIT
        _logger.debug(f'rate limit for {self.resource_pool}: {requests_per_min} RPM')
        self.secs_per_request = 1 / (requests_per_min / 60)

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('request-rate:')

    async def _main_loop(self) -> None:
        last_request_ts = 0.0
        while True:
            item = await self.queue.get()
            if item.num_retries > 0:
                self.total_retried += 1
            now = time.monotonic()
            wait_duration = 0.0
            if item.retry_after is not None:
                wait_duration = item.retry_after - now
            if now - last_request_ts < self.secs_per_request:
                wait_duration = max(wait_duration, self.secs_per_request - (now - last_request_ts))
            if wait_duration > 0:
                _logger.debug(f'waiting for {wait_duration} for {self.resource_pool}')
                await asyncio.sleep(wait_duration)

            last_request_ts = time.monotonic()
            if item.num_retries > 0:
                # the last request encountered some problem: retry it synchronously, to wait for the problem to pass
                _logger.debug(f'retrying request for {self.resource_pool}: #retries={item.num_retries}')
                await self._exec(item.request, item.exec_ctx, item.num_retries, is_task=False)
                _logger.debug(f'retried request for {self.resource_pool}: #retries={item.num_retries}')
            else:
                _logger.debug(f'creating task for {self.resource_pool}')
                self.num_in_flight += 1
                task = asyncio.create_task(self._exec(item.request, item.exec_ctx, item.num_retries, is_task=True))
                self.dispatcher.register_task(task)

    async def _exec(self, request: FnCallArgs, exec_ctx: ExprEvalCtx, num_retries: int, is_task: bool) -> None:
        assert all(not row.has_val[request.fn_call.slot_idx] for row in request.rows)
        assert all(not row.has_exc(request.fn_call.slot_idx) for row in request.rows)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            pxt_fn = request.fn_call.fn
            assert isinstance(pxt_fn, func.CallableFunction)
            _logger.debug(
                f'scheduler {self.resource_pool}: '
                f'start evaluating slot {request.fn_call.slot_idx}, batch_size={len(request.rows)}'
            )
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
            _logger.debug(
                f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} '
                f'in {end_ts - start_ts}, batch_size={len(request.rows)}'
            )
            self.dispatcher.dispatch(request.rows, exec_ctx)

        except Exception as exc:
            _logger.exception(f'exception for {self.resource_pool}: type={type(exc)}\n{exc}')
            if hasattr(exc, 'response') and hasattr(exc.response, 'headers'):
                _logger.debug(f'scheduler {self.resource_pool}: exception headers: {exc.response.headers}')
            is_retriable, retry_after = is_retriable_error(exc)
            if is_retriable and num_retries < self.MAX_RETRIES:
                retry_delay = self._compute_retry_delay(num_retries, retry_after)
                _logger.debug(f'scheduler {self.resource_pool}: retrying after {retry_delay}')
                now = time.monotonic()
                # put the request back in the queue right away, which prevents new requests from being generated until
                # this one succeeds or exceeds its retry limit
                self.queue.put_nowait(self.QueueItem(request, num_retries + 1, exec_ctx, retry_after=now + retry_delay))
                return

            # record the exception
            self.total_errors += 1
            _, _, exc_tb = sys.exc_info()
            for row in request.rows:
                row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc(request.rows, request.fn_call.slot_idx, exc_tb, exec_ctx)
        finally:
            _logger.debug(
                f'Scheduler stats: #in-flight={self.num_in_flight} #requests={self.total_requests}, '
                f'#retried={self.total_retried} #errors={self.total_errors}'
            )
            if is_task:
                self.num_in_flight -= 1

    def _compute_retry_delay(self, num_retries: int, retry_after: float | None = None) -> float:
        """
        Calculate exponential backoff delay for rate limit errors.

        Args:
            retry_count: Number of retries attempted (0-based)
            retry_after: Suggested delay from Retry-After header

        Returns:
            Delay in seconds
        """
        if retry_after is not None and retry_after > 0:
            # Use server-suggested delay, but cap it at max_delay
            return max(min(retry_after, self.MAX_RETRY_DELAY), self.BASE_RETRY_DELAY)
        else:
            return exponential_backoff(num_retries, max_delay=self.MAX_RETRY_DELAY)


# all concrete Scheduler subclasses that implement matches()
SCHEDULERS = [RateLimitsScheduler, RequestRateScheduler]

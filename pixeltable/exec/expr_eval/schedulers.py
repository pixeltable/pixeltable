from __future__ import annotations

import abc
import asyncio
import datetime
import logging
import sys
from dataclasses import dataclass
from typing import Any, Optional, Awaitable, Union

from pixeltable import env
from .globals import Scheduler, FnCallArgs, Dispatcher

_logger = logging.getLogger('pixeltable')


class RateLimitsScheduler(Scheduler):
    """
    Scheduler for FunctionCalls with a RateLimitsInfo pool

    TODO:
    - limit the number of in-flight requests based on the open file limit
    """
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
            # prioritize by number of retries
            return self.num_retries > other.num_retries

    resources: list[str]  # names of resources in pool_info
    resource_pool: str
    queue: asyncio.PriorityQueue[QueueItem]  # prioritizes retries
    loop_task: asyncio.Task
    dispatcher: Dispatcher

    # scheduling-related state
    pool_info: env.RateLimitsInfo
    est_usage: dict[str, Union[int, float]]  # value per resource; accumulated estimates since the last util. report

    num_in_flight: int  # unfinished tasks
    request_completed: asyncio.Event

    total_reported_usage: dict[str, Union[int, float]]  # value per resource; as reported by the API
    total_retried: int

    TIME_FORMAT = '%H:%M.%S %f'
    MAX_RETRIES = 10

    def __init__(self, resources: list[str], resource_pool: str, dispatcher: Dispatcher):
        self.resources = resources
        self.resource_pool = resource_pool
        self.queue = asyncio.PriorityQueue()
        self.dispatcher = dispatcher
        self.loop_task = asyncio.create_task(self._main_loop())
        self.dispatcher.tasks.add(self.loop_task)
        self.pool_info = env.Env.get().get_resource_pool_info(self.resource_pool)
        assert isinstance(self.pool_info, env.RateLimitsInfo)
        self.est_usage = {r: 0 for r in self.resources}
        self.num_in_flight = 0
        self.request_completed = asyncio.Event()
        self.total_reported_usage = {r: 0 for r in self.resources}
        self.total_retried = 0

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
            if not self.pool_info.is_initialized():
                # wait for a single request to get rate limits
                _logger.debug(f'initializing rate limits for {self.resource_pool}')
                await self._exec(item.request, item.num_retries, is_task=False)
                item = None
                continue

            # check rate limits
            request_resources = self._get_request_resources(item.request)
            limits_info = self._check_resource_limits(request_resources)
            if limits_info is not None:
                # wait for capacity to free up
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
                    # force re-acquisition of rate limits before making any scheduling decisions
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
            task.add_done_callback(self.dispatcher.tasks.discard)
            item = None

    def _check_resource_limits(self, request_resources: dict[str, int]) -> Optional[env.RateLimitInfo]:
        """Returns the most underprovisioned resource, or None if all resources are within limits"""
        for resource, usage in request_resources.items():
            # 0.05: leave some headroom, we don't have perfect information
            info = self.pool_info.resource_limits[resource]
            if info.remaining - self.est_usage[resource] - usage < 0.05 * info.limit:
                return info
        return None

    @abc.abstractmethod
    def _get_request_resources(self, request: FnCallArgs) -> dict[str, int]:
        """Return estimated resource usage for the given request"""
        pass

    @abc.abstractmethod
    def _record_usage(self, response: dict) -> None:
        """Update total_reported_usage with the resource usage reported in response"""
        pass

    @abc.abstractmethod
    def _get_retry_delay(self, exc: Exception) -> Optional[float]:
        """Returns number of seconds to wait before retry, or None if not retryable"""
        pass

    async def _exec(self, request: FnCallArgs, num_retries: int, is_task: bool) -> None:
        assert not request.row.has_val[request.fn_call.slot_idx] and not request.row.has_exc(request.fn_call.slot_idx)

        try:
            start_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: start evaluating slot {request.fn_call.slot_idx}')
            result = await request.fn_call.fn.aexec(*request.args, **request.kwargs)
            request.row[request.fn_call.slot_idx] = result
            end_ts = datetime.datetime.now(tz=datetime.timezone.utc)
            _logger.debug(f'scheduler {self.resource_pool}: evaluated slot {request.fn_call.slot_idx} in {end_ts - start_ts}')
            self._record_usage(result)

            # purge accumulated resource usage, now that we have a new report
            self.est_usage = {r: 0 for r in self.resources}

            self.dispatcher.dispatch([request.row])
        except Exception as exc:
            if num_retries < self.MAX_RETRIES:
                retry_delay = self._get_retry_delay(exc)
                if retry_delay is not None:
                    await asyncio.sleep(retry_delay)
                    self.queue.put_nowait(self.QueueItem(request, num_retries + 1))
                    return
            # TODO: update resource limits reported in exc.response.headers, if present

            # record the exception
            _, _, exc_tb = sys.exc_info()
            request.row.set_exc(request.fn_call.slot_idx, exc)
            self.dispatcher.dispatch_exc([request.row], request.fn_call.slot_idx, exc_tb)
        finally:
            _logger.debug(
                f'Scheduler stats: reported_usage={self.total_reported_usage}, total_retried={self.total_retried}')
            if is_task:
                self.num_in_flight -= 1
                self.request_completed.set()


class AnthropicScheduler(RateLimitsScheduler):
    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(['requests', 'input_tokens', 'output_tokens'], resource_pool, dispatcher)

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('anthropic:')

    def _get_request_resources(self, request: FnCallArgs) -> dict[str, int]:
        """Return estimated resource usage for the given request; tuple contains one value per resource"""
        max_tokens = request.fn_call.get_param_value('max_tokens', request.row)
        return {'requests': 1, 'input_tokens': 1, 'output_tokens': max_tokens}

    def _record_usage(self, response: dict) -> None:
        self.total_reported_usage['requests'] += 1
        self.total_reported_usage['input_tokens'] += response['usage']['input_tokens']
        self.total_reported_usage['output_tokens'] += response['usage']['output_tokens']

    def _get_retry_delay(self, exc: Exception) -> Optional[float]:
        import anthropic

        # deal with timeouts separately, they don't come with headers
        if isinstance(exc, anthropic.APITimeoutError):
            return 1.0

        if not isinstance(exc, anthropic.APIStatusError):
            return None
        _logger.debug(f'headers={exc.response.headers}')
        should_retry_str = exc.response.headers.get('x-should-retry', '')
        if should_retry_str.lower() != 'true':
            return None
        retry_after_str = exc.response.headers.get('retry-after', '1')
        return int(retry_after_str)


class OpenAIScheduler(RateLimitsScheduler):

    import openai
    retryable_errors = (
        openai.RateLimitError, openai.APITimeoutError, openai.UnprocessableEntityError, openai.InternalServerError
    )
    import tiktoken
    token_encoder = tiktoken.get_encoding('cl100k_base')

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        super().__init__(['requests', 'tokens'], resource_pool, dispatcher)

    @classmethod
    def matches(cls, resource_pool: str) -> bool:
        return resource_pool.startswith('openai:')

    def _get_request_resources(self, request: FnCallArgs) -> dict[str, int]:
        msgs = request.fn_call.get_param_value('messages', request.row)
        max_tokens = request.fn_call.get_param_value('max_tokens', request.row)
        n = request.fn_call.get_param_value('n', request.row)
        completion_tokens = n * max_tokens

        num_tokens = 0
        for message in msgs:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(self.token_encoder.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens -= 1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return {'requests': 1, 'tokens': num_tokens + completion_tokens}

    def _record_usage(self, response: dict) -> None:
        self.total_reported_usage['requests'] += 1
        if 'usage' in response:
            self.total_reported_usage['tokens'] += response['usage']['total_tokens']

    def _get_retry_delay(self, exc: Exception) -> Optional[float]:
        import openai

        if not isinstance(exc, self.retryable_errors):
            return None
        assert isinstance(exc, openai.APIError)
        if hasattr(exc, 'response') and exc.response is not None:
            _logger.debug(f'headers={exc.response.headers}')
        return 1.0


# all concrete Scheduler subclasses that implement matches()
SCHEDULERS = [AnthropicScheduler, OpenAIScheduler]

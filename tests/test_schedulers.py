# type: ignore

import asyncio
import collections
import datetime
import time
from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import pixeltable as pxt
from pixeltable import env, func
from pixeltable.env import RateLimitsInfo
from pixeltable.exec.expr_eval.globals import Dispatcher, ExprEvalCtx, FnCallArgs
from pixeltable.exec.expr_eval.schedulers import RateLimitsScheduler

_POOL_NAME = 'rate-limits:test-scheduler:default'
_REQUEST_LIMIT = 20


class DummyError(Exception):
    pass


class _ProviderSimulator:
    """Simulates a rate-limited API provider (like OpenAI) using a sliding window.

    Tracks request timestamps in a deque. On each new request, entries older than the
    window are evicted. If the window is already at capacity after eviction, the request
    is rejected (simulating a 429). This models real per-second rate limits more
    faithfully than a simple in-flight counter.

    All access happens on a single asyncio event loop thread (the scheduler's), so no
    locking is needed.
    """

    def __init__(self, max_requests: int, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: collections.deque[float] = collections.deque()
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def enter(self) -> int:
        """Attempt to admit a request into the sliding window.

        Returns the current count of requests in the window & max_request count on success.
        Raises DummyError on rejection (simulated 429).
        """
        now = time.monotonic()
        self._evict(now)
        self.total_calls += 1
        current = len(self._timestamps) + 1
        if current > self.max_requests:
            self.rejections += 1
            raise DummyError(
                f'simulated 429: {current} requests in {self.window_seconds}s window exceeds limit {self.max_requests}'
            )
        self._timestamps.append(now)
        self.peak = max(self.peak, len(self._timestamps))
        return self.max_requests, len(self._timestamps)

    def reset(self) -> None:
        self._timestamps.clear()
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0


_provider = _ProviderSimulator(max_requests=_REQUEST_LIMIT)


def _get_request_resources(text: str) -> dict[str, int]:
    """Estimate resource usage for a single request, dummy to match OpenAI's x-ratelimit-* headers."""
    return {'requests': 1}


class _TestRateLimitsInfo(RateLimitsInfo):
    """RateLimitsInfo with known, controllable limits and no real HTTP headers."""

    def __init__(self, get_request_resources: Callable[..., dict[str, int]]) -> None:
        super().__init__(get_request_resources)

    def record_exc(self, request_ts: datetime.datetime, exc: Exception) -> None:
        with self._lock:
            self.has_exc = True


@pxt.udf(resource_pool=_POOL_NAME, is_deterministic=False)
async def _scheduler_test_udf(text: str) -> str:
    """Fake UDF that simulates a rate-limited API provider.

    Uses _provider's sliding window to enforce a per-second request rate limit and
    reports request count back to the scheduler via record(), the same
    way real provider UDFs (e.g. openai.chat_completions) do.
    """
    pool_info = env.Env.get().get_resource_pool_info(
        _POOL_NAME, lambda: _TestRateLimitsInfo(_get_request_resources)
    )

    # Sliding window check -- raises DummyError on rejection (simulated 429)
    request_limit, request_count = _provider.enter()

    # Report rate limits back, mimicking response headers from a real provider.
    # Like OpenAI's x-ratelimit-* headers, we report the provider's actual window
    # capacity and remaining slots so the scheduler can pace itself accurately.
    # reset_at is 1s from now (matching the window period) so that
    # estimated_resource_refill_delay computes a fast refill rate.
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    reset_at = now + datetime.timedelta(seconds=1)
    pool_info.record(
        request_ts=now,
        reset_exc=True,
        requests=(request_limit, request_limit - request_count, reset_at),
    )
    return f'ok:{text}'


class TestSchedulers:
    def test_rate_limits_scheduler_exception_before_pool(self) -> None:
        """Test that RateLimitsScheduler properly handles exceptions if pool_info is not yet set."""

        async def run_test():
            mock_dispatcher = Mock(spec=Dispatcher)
            mock_dispatcher.register_task = Mock()
            mock_dispatcher.dispatch_exc = Mock()

            scheduler = RateLimitsScheduler('rate-limits:test', mock_dispatcher)

            # Create a mock function that will throw an exception
            mock_fn = Mock(spec=func.CallableFunction)
            mock_fn.aexec = AsyncMock(side_effect=DummyError('Test exception'))
            mock_fn.signature = Mock()
            mock_fn.signature.system_parameters = {'_runtime_ctx'}

            mock_fn_call = Mock()
            mock_fn_call.fn = mock_fn
            mock_fn_call.slot_idx = 0
            mock_fn_call.get_param_values = Mock(return_value=[{}])

            mock_row = Mock()
            mock_row.has_val = [False]
            mock_row.has_exc = Mock(return_value=False)
            mock_row.set_exc = Mock()
            mock_row.__setitem__ = Mock()

            mock_request = Mock(spec=FnCallArgs)
            mock_request.fn_call = mock_fn_call
            mock_request.rows = [mock_row]
            mock_request.row = mock_row
            mock_request.is_batched = False
            mock_request.args = []
            mock_request.kwargs = {}

            mock_ctx = Mock(spec=ExprEvalCtx)

            await scheduler._exec(mock_request, mock_ctx, 0, is_task=False)

            # Verify that the exception was recorded
            assert mock_row.set_exc.called
            assert mock_dispatcher.dispatch_exc.called

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_test())
        finally:
            loop.close()

    def test_rate_limits_scheduler(self, uses_db: None) -> None:
        """Blackbox test: the scheduler respects rate limits reported by the provider.

        The _ProviderSimulator enforces a sliding window rate limit matching the reported
        request limit. The UDF reports accurate rate limit headers back to the scheduler
        (like OpenAI does), so the scheduler should pace itself and complete all rows
        without triggering 429s.

        TODO: add a gemini-style provider simulator that does not return rate limit headers,
        forcing the scheduler to rely on 429 retry recovery.
        """
        num_rows = 200

        _provider.reset()

        try:
            t = pxt.create_table('test_scheduler', {'text': pxt.String})
            t.add_computed_column(result=_scheduler_test_udf(t.text))

            rows = [{'text': f'row-{i}'} for i in range(num_rows)]
            status = t.insert(rows)

            # All rows must succeed
            assert status.num_excs == 0, f'{status.num_excs} rows failed permanently'

            # Every row was processed exactly once (no retries needed)
            assert _provider.total_calls == num_rows, (
                f'total_calls ({_provider.total_calls}) != num_rows ({num_rows}); rejections={_provider.rejections}'
            )

            # No 429s occurred -- the scheduler paced itself using reported rate limits
            assert _provider.rejections == 0, (
                f'{_provider.rejections} simulated 429s occurred; scheduler failed to pace requests '
                f'(total_calls={_provider.total_calls}, peak={_provider.peak})'
            )

            # The sliding window peak should not exceed the provider's window capacity
            assert _provider.peak <= _provider.max_requests, (
                f'peak requests in window {_provider.peak} exceeded provider limit {_provider.max_requests}'
            )
        finally:
            env.Env.get()._resource_pool_info.pop(_POOL_NAME, None)

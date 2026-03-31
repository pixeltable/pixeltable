# type: ignore

import asyncio
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
_REQUEST_LIMIT = 40


class DummyError(Exception):
    pass


class _ProviderSimulator:
    """Simulates a rate-limited API provider using linear request-budget refill.

    Internally scales the requested rate (max_requests per refill_seconds) into a
    larger capacity window, similar to how real providers report limits (e.g., OpenAI
    reports 10000 RPM rather than 167 per second). This keeps the effective rate
    identical while giving the scheduler enough headroom in its in-flight tracking
    so that target_remaining never exceeds the reported limit.

    The budget refills linearly at the same effective rate. This matches the linear
    refill model assumed by estimated_resource_refill_delay in RateLimitInfo.

    All access happens on a single asyncio event loop thread (the scheduler's), so no
    locking is needed.
    """


    def __init__(self, max_requests: int, refill_seconds: float = 1.0) -> None:
        # Scale into a larger window while preserving the effective rate.
        # E.g. 20 req/1s becomes 300 req/15s -- same 20 req/s rate.
        self.max_requests = max_requests
        self.refill_seconds = refill_seconds
        self._available: float = float(self.max_requests)
        self._last_refill: float = time.monotonic()
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0
        self._used_since_refill = 0  # track peak within a refill window

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        refill = elapsed * (self.max_requests / self.refill_seconds)
        self._available = min(self.max_requests, self._available + refill)
        self._last_refill = now
        if self._available >= self.max_requests:
            self._used_since_refill = 0

    def enter(self) -> tuple[int, int]:
        """Attempt to consume one request from the budget.

        Returns (max_requests, used_count) on success where used_count is
        max_requests - remaining requests (floored to int).
        Raises DummyError on rejection (simulated 429).
        """
        now = time.monotonic()
        self._refill(now)
        self.total_calls += 1
        if self._available < 1.0:
            self.rejections += 1
            remaining = max(0, int(self._available))
            raise DummyError(
                f'simulated 429: no requests available (remaining={remaining}, limit={self.max_requests})'
            )
        self._available -= 1.0
        self._used_since_refill += 1
        self.peak = max(self.peak, self._used_since_refill)
        remaining = max(0, int(self._available))
        used = self.max_requests - remaining
        return self.max_requests, used

    def reset(self) -> None:
        self._available = float(self.max_requests)
        self._last_refill = time.monotonic()
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0
        self._used_since_refill = 0


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

    Uses _provider's linear request-budget to enforce a per-second request rate limit
    and reports resource usage back to the scheduler via record(), the same way real
    provider UDFs (e.g. openai.chat_completions) do.
    """
    pool_info = env.Env.get().get_resource_pool_info(
        _POOL_NAME, lambda: _TestRateLimitsInfo(_get_request_resources)
    )

    # Token bucket check -- raises DummyError on rejection (simulated 429)
    request_limit, request_count = _provider.enter()
    await asyncio.sleep(0.05)

    # Report rate limits back, mimicking response headers from a real provider.
    # Like OpenAI's x-ratelimit-* headers, we report the provider's actual bucket
    # capacity and remaining slots so the scheduler can pace itself accurately.
    # reset_at must be proportional to how many requests need refilling so that
    # estimated_resource_refill_delay (which assumes linear refill) computes
    # the correct constant refill rate of max_requests/refill_seconds.
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    time_to_full = request_count * _provider.refill_seconds / request_limit if request_count > 0 else 0.0
    reset_at = now + datetime.timedelta(seconds=time_to_full)
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

        The _ProviderSimulator enforces a linear request-budget rate limit matching the
        reported request limit. The UDF reports accurate rate limit headers back to the
        scheduler (like OpenAI does), so the scheduler should pace itself and complete
        all rows without triggering 429s.

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

            # The request budget peak should not exceed the provider's capacity
            assert _provider.peak <= _provider.max_requests, (
                f'peak requests {_provider.peak} exceeded provider limit {_provider.max_requests}'
            )
        finally:
            env.Env.get()._resource_pool_info.pop(_POOL_NAME, None)

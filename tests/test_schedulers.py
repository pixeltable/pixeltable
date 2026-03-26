# type: ignore

import asyncio
import datetime
import math
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
    """Simulates the state an API provider like OpenAI tracks server-side: how many requests
    are currently in-flight, and whether to reject new ones when capacity is exceeded.

    All access happens on a single asyncio event loop thread (the scheduler's), so no locking
    is needed.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.in_flight = 0
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0

    def enter(self) -> int:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.total_calls += 1
        return self.in_flight

    def reject(self) -> None:
        self.rejections += 1

    def exit(self) -> None:
        self.in_flight -= 1

    def reset(self) -> None:
        self.in_flight = 0
        self.peak = 0
        self.total_calls = 0
        self.rejections = 0


_provider = _ProviderSimulator(capacity=5)


class _TestRateLimitsInfo(RateLimitsInfo):
    """RateLimitsInfo with known, controllable limits and no real HTTP headers."""

    _request_limit: int

    def __init__(self, request_limit: int) -> None:
        super().__init__(get_request_resources=lambda: {'requests': 1})
        self._request_limit = request_limit

    def record_exc(self, request_ts: datetime.datetime, exc: Exception) -> None:
        with self._lock:
            self.has_exc = True

    def get_retry_delay(self, exc: Exception, attempt: int) -> float | None:
        if isinstance(exc, DummyError):
            return 0.1
        return None


@pxt.udf(resource_pool=_POOL_NAME, is_deterministic=False)
async def _scheduler_test_udf(text: str) -> str:
    """Fake UDF that simulates a rate-limited API.

    Tracks concurrency via _provider, enforces its capacity cap, and reports rate
    limits back to the scheduler via record() -- the same way real provider UDFs
    (e.g. openai.chat_completions) do.
    """
    pool_info = env.Env.get().get_resource_pool_info(_POOL_NAME, lambda: _TestRateLimitsInfo(_REQUEST_LIMIT))

    current = _provider.enter()
    try:
        if current > _provider.capacity:
            _provider.reject()
            raise DummyError(f'simulated 429: {current} in-flight exceeds capacity {_provider.capacity}')

        await asyncio.sleep(0.05)

        # Report rate limits back, mimicking response headers from a real provider.
        # reset_at is 1s from now so that estimated_resource_refill_delay computes a
        # fast refill rate; with a far-future reset the refill rate approaches 0 and
        # the scheduler sleeps for hours.
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        reset_at = now + datetime.timedelta(seconds=1)
        pool_info.record(
            request_ts=now, reset_exc=True, requests=(pool_info._request_limit, pool_info._request_limit - 1, reset_at)
        )
        return f'ok:{text}'
    finally:
        _provider.exit()


def _expected_max_concurrency(request_limit: int, cost_per_request: int = 1) -> int:
    """Compute the maximum concurrent requests the scheduler should allow.

    The scheduler fires a new request when:
        remaining >= ceil(limit * TARGET_FRACT + cost + est_in_flight)

    After the synchronous bootstrap, remaining = limit - 1.  Each dispatched
    request adds `cost_per_request` to est_in_flight.  The scheduler stops
    dispatching when est_in_flight pushes the target above remaining.
    """
    fract = env.TARGET_RATE_LIMIT_RESOURCE_FRACT
    headroom = math.ceil(request_limit * fract)
    available_after_bootstrap = request_limit - 1
    budget = available_after_bootstrap - headroom - cost_per_request
    if budget < 0:
        return 1
    return budget // cost_per_request + 1


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

    def test_rate_limits_scheduler_429_recovery(self, uses_db: None) -> None:
        """Blackbox test: the scheduler retries and recovers from simulated 429 errors.

        The _ProviderSimulator enforces a hard concurrency cap (capacity=5) that is deliberately
        lower than what the scheduler's rate-limit math allows (~18 for limit=20).  This
        guarantees some calls get rejected with a simulated 429.  The scheduler must retry
        those calls and eventually complete all rows without permanent failures.
        """
        num_rows = 30

        _provider.reset()

        try:
            t = pxt.create_table('test_scheduler_429', {'text': pxt.String})
            t.add_computed_column(result=_scheduler_test_udf(t.text))

            rows = [{'text': f'row-{i}'} for i in range(num_rows)]
            status = t.insert(rows)

            # All rows must succeed despite 429s
            assert status.num_excs == 0, f'{status.num_excs} rows failed permanently'

            # 429s must have actually occurred (otherwise this test isn't exercising retry logic)
            assert _provider.rejections > 0, (
                'no simulated 429s occurred; test is not exercising retry recovery '
                f'(total_calls={_provider.total_calls}, peak={_provider.peak})'
            )

            # More calls than rows proves retries happened
            assert _provider.total_calls > num_rows, (
                f'total_calls ({_provider.total_calls}) should exceed num_rows ({num_rows}) due to retries'
            )

            # The scheduler should still respect its own calculated concurrency limit
            expected_max = _expected_max_concurrency(_REQUEST_LIMIT)
            assert _provider.peak <= expected_max, (
                f'peak in-flight {_provider.peak} exceeded expected max {expected_max} (request_limit={_REQUEST_LIMIT})'
            )
        finally:
            env.Env.get()._resource_pool_info.pop(_POOL_NAME, None)

# type: ignore

from typing import Callable

import pytest

import pixeltable as pxt
from pixeltable.runtime import get_runtime
from pixeltable.utils import fault_injection
from pixeltable.utils.fault_injection import FaultLocation
from tests.fault_injection import ExceptionFault
from tests.utils import aws_client_error, pxt_raises

pytestmark = pytest.mark.db_roots('local', reason='fault-injection test of the rate-limit scheduler internals')


class DummyError(Exception):
    pass


@pxt.udf(resource_pool='rate-limits:test')
async def _rate_limited_udf(x: int) -> int:  # noqa: RUF029
    return x + 1


@pxt.udf
async def _model_udf(x: int, *, model_id: str) -> int:  # noqa: RUF029
    fault_injection.process_fault(FaultLocation.TEST)
    return x + 1


@_model_udf.resource_pool
def _(model_id: str) -> str:
    return f'request-rate:test:{model_id}'


class TestSchedulers:
    def test_rate_limits_scheduler_exception_before_pool(self, uses_db: None, fault_injection: None) -> None:
        """
        The very first function evaluation raises an error in RateLimitsScheduler. The scheduler correctly identifies
        that pool_info is not yet available, and doesn't attempt to use it to determine the retry delay for this error.
        Before it was fixed, this flow resulted in an AssertionError.
        """
        t = pxt.create_table('test_rate_limits', {'x': pxt.Int | None})
        t.add_computed_column(y=_rate_limited_udf(t.x))

        fault = ExceptionFault(DummyError('Non-retriable error'))
        get_runtime().fault_manager.inject_fault(FaultLocation.SCHEDULER_RATE_LIMITS_AEXEC, fault)

        with pxt_raises(pxt.ErrorCode.UNSUPPORTED_OPERATION, match='DummyError'):
            t.insert([{'x': 1}])

        fault.assert_count(1)

    @pytest.mark.parametrize(
        ('error', 'is_retried'),
        [
            (lambda: aws_client_error('ThrottlingException', 429, 'Please slow down'), True),
            (lambda: pxt.ExternalServiceError(pxt.ErrorCode.PROVIDER_ERROR, 'Server processing failed'), True),
            (
                lambda: pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_ERROR, "{'code': 429, 'message': 'slow down'}", provider='nebius'
                ),
                True,
            ),
            (lambda: pxt.ExternalServiceError(pxt.ErrorCode.RATE_LIMITED, 'Slow down', retry_after=1), True),
            (lambda: pxt.ExternalServiceError(pxt.ErrorCode.PROVIDER_TIMEOUT, 'Job timed out after 600s'), False),
            (
                lambda: pxt.ExternalServiceError(
                    pxt.ErrorCode.PROVIDER_BAD_REQUEST, 'Too many requests in one batch: 120 (the maximum is 100)'
                ),
                False,
            ),
        ],
    )
    def test_request_rate_scheduler_retries(
        self, uses_db: None, fault_injection: None, error: Callable[[], Exception], is_retried: bool
    ) -> None:
        t = pxt.create_table('test_request_rate', {'x': pxt.Int | None})
        t.add_computed_column(y=_model_udf(t.x, model_id='vendor.model-v1:0'))

        fault = ExceptionFault(error())
        get_runtime().fault_manager.inject_fault(FaultLocation.TEST, fault)
        status = t.insert([{'x': 1}], on_error='ignore')
        fault.assert_count(1)
        assert status.num_excs == (0 if is_retried else 1)
        assert t.select(t.y).collect()['y'] == ([2] if is_retried else [None])

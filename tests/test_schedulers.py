# type: ignore

import pytest

import pixeltable as pxt
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.fault_injection import ExceptionFault
from tests.utils import pxt_raises, validate_update_status

pytestmark = pytest.mark.db_roots('local', reason='fault-injection test of the rate-limit scheduler internals')


class DummyError(Exception):
    pass


@pxt.udf(resource_pool='rate-limits:test')
async def _rate_limited_udf(x: int) -> int:  # noqa: RUF029
    return x + 1


@pxt.udf
async def _model_udf(x: int, *, model_id: str) -> int:  # noqa: RUF029
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

    def test_request_rate_scheduler_retries_aws_throttle(self, uses_db: None, fault_injection: None) -> None:
        """AWS throttle is retried."""
        from botocore.exceptions import ClientError

        t = pxt.create_table('test_request_rate', {'x': pxt.Int | None})
        t.add_computed_column(y=_model_udf(t.x, model_id='vendor.model-v1:0'))

        response = {
            'Error': {'Code': 'ThrottlingException', 'Message': 'Please slow down'},
            'ResponseMetadata': {'HTTPStatusCode': 429, 'HTTPHeaders': {}},
        }
        fault = ExceptionFault(ClientError(response, 'InvokeModel'))
        get_runtime().fault_manager.inject_fault(FaultLocation.SCHEDULER_REQUEST_RATE_AEXEC, fault)

        validate_update_status(t.insert([{'x': 1}]), expected_rows=1)
        fault.assert_count(1)
        assert t.select(t.y).collect()['y'] == [2]

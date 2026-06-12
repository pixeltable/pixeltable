# type: ignore

import pytest

import pixeltable as pxt
import pixeltable.exceptions as excs
from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.fault_injection import ExceptionFault


class DummyError(Exception):
    pass


@pxt.udf(resource_pool='rate-limits:test')
async def _rate_limited_udf(x: int) -> int:  # noqa: RUF029
    return x + 1


class TestSchedulers:
    def test_rate_limits_scheduler_exception_before_pool(self, uses_db: None, fault_injection: None) -> None:
        """
        The very first function evaluation raises an error in RateLimitsScheduler. The scheduler correctly identifies
        that pool_info is not yet available, and doesn't attempt to use it to determine the retry delay for this error.
        Before it was fixed, this flow resulted in an AssertionError.
        """
        t = pxt.create_table('test_rate_limits', {'x': pxt.Int})
        t.add_computed_column(y=_rate_limited_udf(t.x))

        fault = ExceptionFault(DummyError('Non-retriable error'))
        get_runtime().fault_manager.inject_fault(FaultLocation.SCHEDULER_RATE_LIMITS_AEXEC, fault)

        with pytest.raises(excs.ExprEvalError, match='DummyError'):
            t.insert([{'x': 1}])

        fault.assert_count(1)

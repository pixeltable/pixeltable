import pytest

from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation
from tests.fault_injection import ExceptionFault


class TestFaultInjection:
    def test_inject_exception(self, fault_injection: None) -> None:
        exc = RuntimeError('injected')
        fault = ExceptionFault(exc)
        get_runtime().fault_manager.inject_fault(FaultLocation.TEST, fault)

        with pytest.raises(RuntimeError, match='injected'):
            get_runtime().fault_manager.process_location(FaultLocation.TEST)

        fault.assert_count(1)

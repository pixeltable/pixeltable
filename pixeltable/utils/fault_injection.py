"""
A lightweight fault injection framework for testing failure and concurrency scenarios in instrumented code paths.

Production code calls `process_fault(loc)` at each `FaultLocation`. In production this is a
no-op (`ProdFaultManager`). In tests, a `TestFaultManager` can register a `Fault` at any
location before triggering the code under test:

- `ExceptionFault`: raises an exception
- `BlockFault`: pauses the thread at the fault point, letting the test inspect or manipulate shared state before calling
`unblock()`
"""

import abc
import logging
import os
from enum import Enum, auto
from threading import Event, Lock
from typing import Any, Protocol

import pixeltable.exceptions as excs

_logger = logging.getLogger('pixeltable')


class FaultLocation(Enum):
    """Instrumented locations in the codebase where faults can be injected."""

    CATALOG_FINALIZE_PENDING_OPS_NON_XACT = auto()
    CATALOG_LOAD_VIEW_OP_EXEC = auto()


class Fault(Protocol):
    recurring: bool

    def __call__(self, loc: FaultLocation) -> Any: ...


class FaultManager(abc.ABC):
    @abc.abstractmethod
    def inject_fault(self, loc: FaultLocation, fault: Fault) -> None: ...

    @abc.abstractmethod
    def process_fault(self, loc: FaultLocation) -> None: ...

    @abc.abstractmethod
    def clear_faults(self) -> None: ...


class ProdFaultManager(FaultManager):
    def inject_fault(self, loc: FaultLocation, fault: Fault) -> None:
        raise excs.Error(excs.ErrorCode.INTERNAL_ERROR, 'Fault injection not allowed outside of tests')

    def process_fault(self, loc: FaultLocation) -> None:
        return

    def clear_faults(self) -> None:
        return


class TestFaultManager(FaultManager):
    _fault_state: dict[FaultLocation, Fault]
    # guards _fault_state for thread-safe access
    _lock: Lock

    def __init__(self) -> None:
        self._fault_state = {}
        self._lock = Lock()

    def inject_fault(self, loc: FaultLocation, fault: Fault) -> None:
        with self._lock:
            if loc in self._fault_state:
                raise AssertionError(f'A fault is already armed at {loc}')
            self._fault_state[loc] = fault

    def process_fault(self, loc: FaultLocation) -> None:
        with self._lock:
            fault = self._fault_state.get(loc)
            if fault is None:
                return
            if not fault.recurring:
                del self._fault_state[loc]
        fault(loc)

    def clear_faults(self) -> None:
        with self._lock:
            self._fault_state.clear()


def create_fault_manager() -> FaultManager:
    if os.environ.get('PYTEST_CURRENT_TEST'):
        _logger.debug('Running in a test, initializing fault manager')
        return TestFaultManager()
    _logger.debug('Not running in a test, fault injection disabled')
    return ProdFaultManager()


class ExceptionFault:
    e: Exception
    _counter: int
    # guards internal state for thread safety
    _lock: Lock

    def __init__(self, e: Exception, recurring: bool = False) -> None:
        self._lock = Lock()
        self._counter = 0
        self.e = e
        self.recurring = recurring

    def __call__(self, loc: FaultLocation) -> None:
        _logger.info(f'Injecting {type(self.e)} at fault location {loc}')
        with self._lock:
            self._counter += 1
            raise type(self.e)(*self.e.args)

    def counter(self) -> int:
        with self._lock:
            return self._counter

    def assert_count(self, expected: int) -> None:
        with self._lock:
            assert self._counter == expected, f'Expected {expected} fault hits, was {self._counter}'


class BlockFault:
    # Note: a FaultManager sometimes can become shared between threads (see Runtime.copy_db_context), and as the result,
    # so can a BlockFault. It's not clear what the desired behavior is in that case, so for now all BlockFaults are
    # non-recurring, meaning, only the first thread to hit one will be blocked. This can be changed in the future if
    # there is a need for a different behavior.

    # an event that is signaled (set) when the fault is reached
    reached: Event
    # an event that is signaled when the blocked thread should proceed
    unblocked: Event

    def __init__(self) -> None:
        self.reached = Event()
        self.unblocked = Event()
        self.recurring = False

    def __call__(self, loc: FaultLocation) -> None:
        _logger.info(f'Blocked at {loc}')
        self.reached.set()
        self.unblocked.wait()
        _logger.info(f'Unblocked at {loc}')

    def block_until_reached(self) -> None:
        _logger.info('Waiting until fault point is reached...')
        self.reached.wait()
        _logger.info('BlockFault reached')

    def unblock(self) -> None:
        _logger.info('Unblocking')
        self.unblocked.set()


def process_fault(loc: FaultLocation) -> None:
    from pixeltable.runtime import get_runtime

    get_runtime().fault_manager.process_fault(loc)

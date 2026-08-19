"""
A lightweight fault injection framework for testing failure and concurrency scenarios in instrumented code paths.

Product code calls `process_fault(loc)` at each `FaultLocation`. In production this is a no-op. Tests that use fault
injection, monkey patch the actual working implementation using the `fault_injection` fixture.

When fault injection is enabled, each thread/Runtime gets a separate `FaultManager`. Faults of the following types can
be injected in the fault manager:

- `ExceptionFault`: raises an exception
- `BlockFault`: pauses the thread at the fault point, letting the test inspect or manipulate shared state before calling
`unblock()`

A fault can carry a `filter` over the kwargs its call site passes, for a location that serves more than one caller.
"""

import copy
import logging
from abc import ABC, abstractmethod
from threading import Event, Lock
from typing import Any, Callable

from pixeltable.utils.fault_injection import FaultLocation

_logger = logging.getLogger('pixeltable_test')


class Fault(ABC):
    """A fault that can be armed at a FaultLocation."""

    # if False, the fault is disarmed by the first hit it applies to
    recurring: bool
    # selects the hits this fault applies to, as a predicate over the call site's kwargs; None means applies to all
    _filter: Callable[[dict[str, Any]], bool] | None

    def __init__(self, recurring: bool, filter: Callable[[dict[str, Any]], bool] | None) -> None:
        self.recurring = recurring
        self._filter = filter

    def applies(self, loc: FaultLocation, kwargs: dict[str, Any]) -> bool:
        """Whether this fault fires for this hit. A fault that does not apply is left armed for a later one.

        Called while the FaultManager lock is held, so an override must not block.
        """
        return self._filter is None or self._filter(kwargs)

    @abstractmethod
    def __call__(self, loc: FaultLocation, kwargs: dict[str, Any]) -> Any: ...


class FaultManager:
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

    def process_location(self, loc: FaultLocation, kwargs: dict[str, Any]) -> None:
        with self._lock:
            fault = self._fault_state.get(loc)
            if fault is None or not fault.applies(loc, kwargs):
                return
            if not fault.recurring:
                del self._fault_state[loc]
        fault(loc, kwargs)


class ExceptionFault(Fault):
    e: Exception
    _counter: int
    # guards internal state for thread safety
    _lock: Lock

    def __init__(
        self, e: Exception, recurring: bool = False, filter: Callable[[dict[str, Any]], bool] | None = None
    ) -> None:
        super().__init__(recurring, filter)
        self._lock = Lock()
        self._counter = 0
        self.e = e

    def __call__(self, loc: FaultLocation, kwargs: dict[str, Any]) -> None:
        _logger.info(f'Injecting {type(self.e)} at fault location {loc}')
        with self._lock:
            self._counter += 1
            raise copy.copy(self.e)

    def counter(self) -> int:
        with self._lock:
            return self._counter

    def assert_count(self, expected: int) -> None:
        with self._lock:
            assert self._counter == expected, f'Expected {expected} fault hits, was {self._counter}'


class BlockFault(Fault):
    # Note: a FaultManager sometimes can become shared between threads (see Runtime.copy_db_context), and as the result,
    # so can a BlockFault. It's not clear what the desired behavior is in that case, so for now all BlockFaults are
    # non-recurring, meaning, only the first thread to hit one will be blocked. This can be changed in the future if
    # there is a need for a different behavior.

    # an event that is signaled (set) when the fault is reached
    reached: Event
    # an event that is signaled when the blocked thread should proceed
    unblocked: Event

    def __init__(self, filter: Callable[[dict[str, Any]], bool] | None = None) -> None:
        super().__init__(recurring=False, filter=filter)
        self.reached = Event()
        self.unblocked = Event()

    def __call__(self, loc: FaultLocation, kwargs: dict[str, Any]) -> None:
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


def create_fault_manager() -> FaultManager:
    return FaultManager()


def process_fault(loc: FaultLocation, **kwargs: Any) -> None:
    from pixeltable.runtime import get_runtime

    get_runtime().fault_manager.process_location(loc, kwargs)

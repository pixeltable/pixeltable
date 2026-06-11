import logging
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pixeltable.utils.fault_injection import FaultLocation

    from .fault_injection import Fault

_logger = logging.getLogger('pixeltable_test')


@dataclass
class _Step:
    thread_id: int
    name: str
    fn: Callable[[], Any]
    gate: Event | None = None
    next_gate: Event | None = None


class MultiThreadedScenario:
    """
    A builder for deterministic multi-threaded test scenarios. Steps are added with
    `then_run` / `then_run_until` and executed in strict sequential order across threads,
    with each step gated on the previous one completing.
    """

    _steps: list[_Step]

    def __init__(self) -> None:
        self._steps = []

    def then_run(self, *, thread_id: int, name: str, fn: Callable[[], Any]) -> 'MultiThreadedScenario':
        """Append a step that runs `fn` synchronously on Thread `thread_id`."""
        self._steps.append(_Step(thread_id=thread_id, name=name, fn=fn))
        return self

    def then_run_until(
        self, *, thread_id: int, name: str, event: Event, fn: Callable[[], Any]
    ) -> 'MultiThreadedScenario':
        """Append a step that runs `fn` on Thread `thread_id`; `event` gates the next step."""
        self._steps.append(_Step(thread_id=thread_id, name=name, fn=fn, next_gate=event))
        return self

    def then_inject_fault(self, *, thread_id: int, loc: 'FaultLocation', fault: 'Fault') -> 'MultiThreadedScenario':
        """Append a step that arms `fault` at `loc` on Thread `thread_id`."""
        from pixeltable.runtime import get_runtime

        return self.then_run(
            thread_id=thread_id,
            name=f'inject fault at {loc.name}',
            fn=lambda: get_runtime().fault_manager.inject_fault(loc, fault),
        )

    def execute(self, timeout: float = 10.0) -> None:
        """Run the scenario. Raises the first exception encountered in any thread."""
        if not self._steps:
            return

        assert self._steps[-1].next_gate is None, (
            f'Scenario ends with an "until" step ("{self._steps[-1].name}"); '
            'the condition event would never be waited on'
        )

        # Fill in absent events and wire the gate chain so step[i].next_gate == step[i+1].gate.
        # "until" steps already have next_gate set; "then" steps get a new one here.
        for step in self._steps:
            if step.next_gate is None:
                step.next_gate = Event()
        first_gate = Event()
        first_gate.set()
        for i, step in enumerate(self._steps):
            if step.gate is None:
                step.gate = first_gate if i == 0 else self._steps[i - 1].next_gate

        # thread id to steps
        thread_work: dict[int, list[_Step]] = {}
        for step in self._steps:
            thread_work.setdefault(step.thread_id, []).append(step)

        # [step name, exception]
        exceptions: list[tuple[str, BaseException]] = []
        exc_lock = Lock()
        abort = Event()
        deadline = time.monotonic() + timeout

        def record_exc(e: BaseException, step: _Step) -> None:
            with exc_lock:
                exceptions.append((step.name, e))

        def run_worker(steps: list[_Step]) -> None:
            for step in steps:
                assert step.gate is not None
                assert step.next_gate is not None
                if abort.is_set():
                    return
                try:
                    _logger.info(f'[{current_thread().name}] waiting for gate: {step.name}')
                    while True:
                        unblocked = step.gate.wait(timeout=0.1)
                        if abort.is_set():
                            return
                        if unblocked:
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'{step.name} timed out waiting on its gate')
                    _logger.info(f'[{current_thread().name}] running: {step.name}')
                    step.fn()
                except BaseException as e:
                    record_exc(e, step)
                    # stop processing this thread's remaining steps, and signal to other threads to do the same
                    abort.set()
                    return
                finally:
                    # always unblock downstream
                    step.next_gate.set()

        workers = [
            Thread(target=run_worker, args=(steps,), name=f'scenario-thread-{tid}', daemon=True)
            for tid, steps in thread_work.items()
        ]

        for w in workers:
            w.start()

        def check_exceptions() -> None:
            with exc_lock:
                if exceptions:
                    _, exc = exceptions[0]
                    raise exc

        for w in workers:
            remaining = deadline - time.monotonic()
            w.join(timeout=max(remaining, 0))
            if w.is_alive():
                abort.set()
                check_exceptions()
                raise TimeoutError(f'Scenario did not finish within {timeout}s')

        check_exceptions()

import logging
import time
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread
from typing import Any, Callable

from pixeltable.runtime import get_runtime
from pixeltable.utils.fault_injection import FaultLocation

from .fault_injection import BlockFault, Fault

_logger = logging.getLogger('pixeltable_test')

_POLL_INTERVAL = 0.500


@dataclass
class _Step:
    thread_id: int
    name: str
    fn: Callable[[], Any]
    gate: Event | None = None
    next_gate: Event | None = None
    poll_condition: Callable[[], bool] | None = None


class MultiThreadedScenario:
    """
    A builder for deterministic multi-threaded test scenarios. Steps are added with
    `then_run` / `then_run_until` and executed in strict sequential order across threads,
    with each step gated on the previous one completing.
    """

    _steps: list[_Step]
    # every BlockFault this scenario refers to, in registration order
    _block_faults: list[BlockFault]

    def __init__(self) -> None:
        self._steps = []
        self._block_faults = []

    def then_run(self, *, thread_id: int, name: str, fn: Callable[[], Any]) -> 'MultiThreadedScenario':
        """Append a step that runs `fn` synchronously on Thread `thread_id`."""
        self._steps.append(_Step(thread_id=thread_id, name=name, fn=fn))
        return self

    def then_run_until(
        self,
        *,
        thread_id: int,
        name: str,
        fn: Callable[[], Any],
        event: Event | None = None,
        poll_condition: Callable[[], bool] | None = None,
    ) -> 'MultiThreadedScenario':
        """Append a step that runs `fn` on Thread `thread_id`, gating the next step on `fn` reaching some point.

        Exactly one of:
        - event: `fn` sets it from the inside, typically a BlockFault parking at a fault point. The next step is
          admitted when it is set.
        - poll_condition: a predicate the scenario evaluates on separate thread, such as a PostgreSQL lock
          blocking. The next step is admitted as soon as a poll observes it true, so the condition must describe a
          state that `fn` stays in until a later step releases it. The step fails if `fn` returns before a poll
          observed the condition.

        Either way the next step is admitted while `fn` is still running, and `fn` runs on past that point.
        """
        assert (event is None) != (poll_condition is None), 'pass exactly one of event, poll_condition'
        self._steps.append(_Step(thread_id=thread_id, name=name, fn=fn, next_gate=event, poll_condition=poll_condition))
        return self

    def then_inject_fault(self, *, thread_id: int, loc: FaultLocation, fault: Fault) -> 'MultiThreadedScenario':
        """Append a step that arms `fault` at `loc` on Thread `thread_id`."""
        self._register_block_fault(fault)
        return self.then_run(
            thread_id=thread_id,
            name=f'inject fault at {loc.name}',
            fn=lambda: get_runtime().fault_manager.inject_fault(loc, fault),
        )

    def then_unblock(self, *, thread_id: int, fault: BlockFault) -> 'MultiThreadedScenario':
        """Append a step that unblocks `fault` on Thread `thread_id`."""
        self._register_block_fault(fault)
        return self.then_run(thread_id=thread_id, name='unblock', fn=fault.unblock)

    def _register_block_fault(self, fault: Fault) -> None:
        if isinstance(fault, BlockFault) and not any(f is fault for f in self._block_faults):
            self._block_faults.append(fault)

    def _unblock_all(self) -> None:
        for fault in self._block_faults:
            fault.unblock()

    def execute(self, timeout: float = 10.0) -> None:
        """Run the scenario. Raises the first exception encountered in any thread."""
        if not self._steps:
            return

        last_step = self._steps[-1]
        assert last_step.next_gate is None and last_step.poll_condition is None, (
            f'Scenario ends with an "until" step ("{last_step.name}"); its condition would never be waited on'
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

        def fail(e: BaseException, step: _Step) -> None:
            """Record an exception and tear the scenario down, releasing anyone parked at a fault point."""
            with exc_lock:
                exceptions.append((step.name, e))
            abort.set()
            self._unblock_all()

        def start_poller(step: _Step) -> None:
            """Start polling `step`'s condition on a dedicated thread. Open `step`'s gate when it holds."""
            assert step.poll_condition is not None
            assert step.next_gate is not None
            # give up a little before execute() stops joining workers, so that this step's specific failure is
            # the one reported rather than the generic "scenario timed out"
            poll_deadline = deadline - 2 * _POLL_INTERVAL

            def poll() -> None:
                try:
                    # the gate also gets set when fn returns, which is the signal to stop polling
                    while not abort.is_set() and not step.next_gate.is_set():
                        try:
                            if step.poll_condition():
                                _logger.info(f'poll condition met: {step.name}')
                                return
                        except BaseException as e:
                            fail(e, step)
                            return
                        if time.monotonic() > poll_deadline:
                            fail(TimeoutError(f'{step.name}: poll condition not met'), step)
                            return
                        time.sleep(_POLL_INTERVAL)
                finally:
                    # every exit, success or failure, opens the next gate
                    step.next_gate.set()

            Thread(target=poll, name=f'poll-{step.thread_id}', daemon=True).start()

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
                    if step.poll_condition is not None:
                        start_poller(step)
                    step.fn()
                    if step.poll_condition is not None and not step.next_gate.is_set():
                        raise AssertionError(f'{step.name}: returned before its poll condition was met')
                except BaseException as e:
                    fail(e, step)
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
                self._unblock_all()
                check_exceptions()
                raise TimeoutError(f'Scenario did not finish within {timeout}s')

        check_exceptions()

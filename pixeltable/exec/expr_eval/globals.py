import abc
import asyncio
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

from pixeltable import exprs


@dataclass
class FnCallArgs:
    fn_call: exprs.FunctionCall
    row: exprs.DataRow
    args: list[Any]
    kwargs: dict[str, Any]


class Scheduler(abc.ABC):
    """
    Base class for schedulers. A scheduler executes FunctionCalls against a limited resource pool.

    Expected behavior:
    - all created tasks must be recorded in dispatcher.tasks
    - schedulers are responsible for aborting execution when a) the task is cancelled or b) when an exception occurred
      elsewhere (indicated by dispatcher.exc_event)
    """
    @abc.abstractmethod
    def submit(self, item: FnCallArgs) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def matches(cls, resource_pool: str) -> bool:
        """Returns True if the scheduler can handle the given resource pool"""
        pass


class Dispatcher(Protocol):
    """Row dispatcher used by Evaluators as a post-processing step after slot materialization"""
    tasks: set[asyncio.Task]
    row_builder: exprs.RowBuilder
    exc_event: asyncio.Event
    schedulers: dict[str, Scheduler]

    def dispatch(self, rows: list[exprs.DataRow]) -> None:
        """Dispatches row slots to the appropriate schedulers; does not block"""
        ...

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType) -> None:
        """Propagates exception in slot_with_exc to all dependent slots and dispatches the rest; does not block"""
        ...


class Evaluator(abc.ABC):
    """
    Base class for expression evaluators. Each DataRow slot is assigned an evaluator, which is responsible for the
    execution of the expression evaluation logic as well as the scheduling/task breakdown of that execution.

    Expected behavior:
    - all created tasks must be recorded in dispatcher.tasks
    - evaluators are responsible for aborting execution when a) the task is cancelled or b) when an exception occurred
      elsewhere (indicated by dispatcher.exc_event)
    """
    dispatcher: Dispatcher
    is_closed: bool

    def __init__(self, dispatcher: Dispatcher):
        self.dispatcher = dispatcher
        self.is_closed = False

    @abc.abstractmethod
    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        """
        Create tasks to evaluate the expression in the given slot for the given rows; must not block.

        """
        pass

    def _close(self) -> None:
        """Close the evaluator; must not block"""
        pass

    def close(self) -> None:
        """Indicates that there may not be any more rows getting scheduled"""
        self.is_closed = True
        self._close()


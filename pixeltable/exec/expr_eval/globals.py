from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Iterable, Optional, Protocol

import numpy as np

from pixeltable import exprs, func


@dataclass
class FnCallArgs:
    """Container for everything needed to execute a FunctionCall against one or more DataRows"""

    fn_call: exprs.FunctionCall
    rows: list[exprs.DataRow]
    # single call
    args: Optional[list[Any]] = None
    kwargs: Optional[dict[str, Any]] = None
    # batch call
    batch_args: Optional[list[list[Optional[Any]]]] = None
    batch_kwargs: Optional[dict[str, list[Optional[Any]]]] = None

    @property
    def pxt_fn(self) -> func.CallableFunction:
        assert isinstance(self.fn_call.fn, func.CallableFunction)
        return self.fn_call.fn

    @property
    def is_batched(self) -> bool:
        return self.batch_args is not None

    @property
    def row(self) -> exprs.DataRow:
        assert len(self.rows) == 1
        return self.rows[0]


class Scheduler(abc.ABC):
    """
    Base class for queueing schedulers. A scheduler executes FunctionCalls against a limited resource pool.

    Expected behavior:
    - all created tasks must be recorded in dispatcher.tasks
    - schedulers are responsible for aborting execution when a) the task is cancelled or b) when an exception occurred
      elsewhere (indicated by dispatcher.exc_event)
    """

    @dataclass(frozen=True)
    class QueueItem:
        """Container of work items for queueing schedulers"""

        request: FnCallArgs
        num_retries: int
        exec_ctx: ExecCtx

        def __lt__(self, other: Scheduler.QueueItem) -> bool:
            # prioritize by number of retries (more retries = higher priority)
            return self.num_retries > other.num_retries

    resource_pool: str
    queue: asyncio.PriorityQueue[QueueItem]  # prioritizes retries
    dispatcher: Dispatcher

    def __init__(self, resource_pool: str, dispatcher: Dispatcher):
        self.resource_pool = resource_pool
        self.queue = asyncio.PriorityQueue()
        self.dispatcher = dispatcher

    def submit(self, item: FnCallArgs, exec_ctx: ExecCtx) -> None:
        self.queue.put_nowait(self.QueueItem(item, 0, exec_ctx))

    @classmethod
    @abc.abstractmethod
    def matches(cls, resource_pool: str) -> bool:
        """Returns True if the scheduler can handle the given resource pool"""
        pass


class Dispatcher(Protocol):
    """
    Row dispatcher used by Evaluators/Schedulers for post-processing after slot materialization and for task management.

    Task management: all tasks need to be registered via register_task()
    Exceptions: evaluators/schedulers need to check exc_event prior to starting long-running (non-interruptible)
        computations
    """

    row_builder: exprs.RowBuilder
    exc_event: asyncio.Event
    schedulers: dict[str, Scheduler]  # key: resource pool id

    def dispatch(self, rows: list[exprs.DataRow], exec_ctx: Any) -> None:
        """Dispatches row slots to the appropriate schedulers; does not block"""
        ...

    def dispatch_exc(self, rows: list[exprs.DataRow], slot_with_exc: int, exc_tb: TracebackType, exec_ctx: Any) -> None:
        """Propagates exception in slot_with_exc to all dependent slots and dispatches the rest; does not block"""
        ...

    def register_task(self, f: asyncio.Task) -> None:
        """Register task with dispatcher for subsequent cleanup; does not block"""
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
    exec_ctx: 'ExecCtx'

    def __init__(self, dispatcher: Dispatcher, exec_ctx: 'ExecCtx') -> None:
        self.dispatcher = dispatcher
        self.is_closed = False
        self.exec_ctx = exec_ctx

    @abc.abstractmethod
    def schedule(self, rows: list[exprs.DataRow], slot_idx: int) -> None:
        """Create tasks to evaluate the expression in the given slot for the given rows; must not block."""

    def _close(self) -> None:
        """Close the evaluator; must not block"""
        pass

    def close(self) -> None:
        """Indicates that there may not be any more rows getting scheduled"""
        self.is_closed = True
        self._close()


class ExecCtx:
    """DataRow-specific state needed by ExprEvalNode"""

    row_builder: exprs.RowBuilder
    slot_evaluators: dict[int, Evaluator]  # key: slot idx
    gc_targets: np.ndarray  # bool per slot; True if this is an intermediate expr (ie, not part of our output)
    eval_ctx: np.ndarray  # bool per slot; EvalCtx.slot_idxs as a mask
    literals: dict[int, Any]  # key: slot idx; value: literal value for this slot; used to pre-populate rows

    def __init__(
        self,
        dispatcher: Dispatcher,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
    ):
        self.row_builder = row_builder
        self.slot_evaluators = {}
        # TODO: only include output_exprs dependencies
        self.gc_targets = np.ones(self.row_builder.num_materialized, dtype=bool)
        # we need to retain all slots that are part of the output
        self.gc_targets[[e.slot_idx for e in self.row_builder.output_exprs]] = False

        output_ctx = self.row_builder.create_eval_ctx(output_exprs, exclude=input_exprs)
        self.literals = {e.slot_idx: e.val for e in output_ctx.exprs if isinstance(e, exprs.Literal)}
        self.eval_ctx = np.zeros(self.row_builder.num_materialized, dtype=bool)
        non_literal_slot_idxs = [e.slot_idx for e in output_ctx.exprs if not isinstance(e, exprs.Literal)]
        self.eval_ctx[non_literal_slot_idxs] = True
        self._init_slot_evaluators(dispatcher, non_literal_slot_idxs)

    def _init_slot_evaluators(self, dispatcher: Dispatcher, target_slot_idxs: list[int]) -> None:
        from .evaluators import DefaultExprEvaluator, FnCallEvaluator, JsonMapperDispatcher

        for slot_idx in target_slot_idxs:
            expr = self.row_builder.unique_exprs[slot_idx]
            if (
                isinstance(expr, exprs.FunctionCall)
                # ExprTemplateFunction and AggregateFunction calls are best handled by FunctionCall.eval()
                and not isinstance(expr.fn, func.ExprTemplateFunction)
                and not isinstance(expr.fn, func.AggregateFunction)
            ):
                self.slot_evaluators[slot_idx] = FnCallEvaluator(expr, dispatcher, self)
            elif isinstance(expr, exprs.JsonMapperDispatch):
                self.slot_evaluators[slot_idx] = JsonMapperDispatcher(expr, dispatcher, self)
            else:
                self.slot_evaluators[slot_idx] = DefaultExprEvaluator(expr, dispatcher, self)

    def init_rows(self, rows: list[exprs.DataRow]) -> None:
        """Pre-populate rows with literals and initialize execution state"""
        for row in rows:
            # set literals before missing_dependents/slots
            for slot_idx, val in self.literals.items():
                row[slot_idx] = val
            row.missing_dependents = np.sum(self.row_builder.dependencies[row.has_val == False], axis=0)
            row.missing_slots = self.eval_ctx & (row.has_val == False)

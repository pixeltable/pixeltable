from __future__ import annotations

import abc
import asyncio
import contextlib
import queue
import threading
from types import TracebackType
from typing import Any, AsyncGenerator, Generator, Iterable, Iterator, Self, TypeVar

import pixeltable.exceptions as excs
import pixeltable.type_system as ts
from pixeltable import exprs, telemetry
from pixeltable.runtime import get_runtime

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext


class ExecNode(abc.ABC):
    """
    Base class of all execution nodes.

    Lifecycle:
    1. The immutable node structure (output_exprs, input node, etc.) can be created incrementally
       (ie, after init(); the planner might need to make adjustments after instance construction)
    2. init_bindings() assumes the immutable structure is complete and sets bind_sources/vars
    3. bind_params()/iter() can then be called repeatedly to execute the same plan with different parameters
    4. _open() initializes per-iteration execution state

    Closing: an iteration can stop before its last batch (eg, a cursor loop that breaks), and its cleanup (eg,
    ColumnRef.release()) has to run right then, not at garbage collection.
    - __aiter__() returns an async generator, which Python doesn't close when its last reference goes away: it
      hands the generator to the event loop, which closes it later, if at all. Callers must therefore close it with
      aclose(), also after an early stop, and an implementation that iterates over self.input must close
      aiter(self.input) the same way, eg, with contextlib.aclosing().
    - __iter__() returns a generator, which CPython closes as soon as its last reference goes away, eg, when the for
      loop over it exits. Callers don't close it explicitly.

    Not thread-safe.
    """

    output_exprs: Iterable[exprs.Expr]
    row_builder: exprs.RowBuilder
    input: ExecNode | None
    flushed_img_slots: list[int]  # idxs of image slots of our output_exprs dependencies
    ctx: ExecContext | None

    # source exprs used to extract Variables; populated by finalize()
    bind_sources: list[exprs.Expr]

    # Variables found in bind_sources; populated by finalize()
    vars: list[exprs.Variable]

    # values bound for this node's parameters (typically Variables); populated by bind_params()
    bound_args: dict[str, Any]

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: ExecNode | None = None,
    ):
        assert all(expr.is_valid for expr in output_exprs)
        self.output_exprs = output_exprs
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.ctx = input.ctx if input is not None else None
        self.bind_sources = []
        self.vars = []
        self.bound_args = {}

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def init_bindings(self) -> None:
        """Populate self.vars from self.bind_sources.

        Subclasses need to override this to set bind_sources first, then call super().init_bindings().
        """
        vars: dict[str, exprs.Variable] = {}
        for v in exprs.Expr.list_subexprs(self.bind_sources, exprs.Variable):
            existing = vars.get(v.name)
            if existing is None:
                vars[v.name] = v
            elif existing.col_type != v.col_type:
                raise AssertionError(
                    f'Variable {v.name!r} appears with conflicting types: {existing.col_type} vs {v.col_type}'
                )
        self.vars = list(vars.values())

    def params(self) -> dict[str, ts.ColumnType]:
        """Return the parameter signature of this node. Valid after init_bindings()."""
        return {v.name: v.col_type for v in self.vars}

    def bind_params(self, args: dict[str, Any]) -> None:
        self.bound_args = {}
        exprs.Expr.prepare_list(self.bind_sources, args, self.bound_args)

    def set_var_slots(self, rows: Iterable[exprs.DataRow]) -> None:
        """Populate Variable slots in rows with the bound values from self.bound_args."""
        for v in self.vars:
            if v.slot_idx is None:
                # parameter-only Variable (eg, the limit/offset on AggregationNode/FilterNode); not
                # materialized per row, just needs to be in bind_sources for value coercion
                continue
            val = self.bound_args[v._bind_name]
            for row in rows:
                row[v.slot_idx] = val

    @abc.abstractmethod
    def __aiter__(self) -> AsyncGenerator[DataRowBatch, None]:  # noqa: PYI058
        """Yield this node's output batches.

        Callers must close the generator with aclose(); see the class docstring.
        """

    def __iter__(self) -> Generator[DataRowBatch, None, None]:
        if asyncio._get_running_loop() is None:
            # if we don't already have a running loop, we can run the async iterator directly and avoid the extra
            # overhead introduced by thread_iter()
            loop = get_runtime().event_loop
            aiter = self.__aiter__()
            step: asyncio.Future[DataRowBatch] | None = None
            try:
                while True:
                    # wrap the next step in a Task, which the finally block can cancel
                    step = asyncio.ensure_future(aiter.__anext__(), loop=loop)
                    batch: DataRowBatch = loop.run_until_complete(step)
                    yield batch
            except StopAsyncIteration:
                pass
            finally:
                # skip the cleanup where run_until_complete() raises:
                # - the loop is closed: Runtime.close() has already closed aiter
                # - another loop is running in this thread (eg, async code closes this generator): the runtime loop's
                #   async generator finalizer schedules aiter.aclose() once aiter is garbage-collected
                if not loop.is_closed() and asyncio._get_running_loop() is None:
                    if step is not None and not step.done():
                        # an exception (eg, KeyboardInterrupt) left run_until_complete() while step was still running,
                        # and aclose() raises for a generator in the middle of __anext__(): cancel step and wait first
                        step.cancel()
                        loop.run_until_complete(asyncio.wait([step]))
                    loop.run_until_complete(aiter.aclose())
        else:
            yield from self.thread_iter()

    _THREAD_QUEUE_SENTINEL = object()

    def thread_iter(self) -> Iterator[DataRowBatch]:
        """Run the async iterator in a new thread with a dedicated event loop, connected via a queue.

        The new thread uses the database context of the thread that makes the first pull. After the first pull, any
        thread can advance the returned iterator.
        """
        # maxsize=2: we want a minimal amount of buffering to allow for some overlap between plan execution and result
        # consumption (but buffering more than that would simply increase memory consumption without additional
        # benefits)
        result_queue: queue.Queue = queue.Queue(maxsize=2)
        producer_task: asyncio.Task | None = None  # set before the first put()
        caller_runtime = get_runtime()
        # carry the ambient instrumentation span across the thread boundary so spans reported on the
        # worker thread nest under it
        hooks_ctx = telemetry.capture_context()

        def run() -> None:
            thread_runtime = get_runtime()
            # the execution needs to happen in the same db context as the caller, but on a new event loop
            thread_runtime.copy_db_context(caller_runtime)
            hooks_token = telemetry.restore_context(hooks_ctx)

            async def produce() -> None:
                nonlocal producer_task
                producer_task = asyncio.current_task()
                async with contextlib.aclosing(aiter(self)) as batches:
                    async for batch in batches:
                        result_queue.put(batch)

            # asyncio.Runner gives this thread the same loop teardown that asyncio.run() performs
            with asyncio.Runner() as runner:
                try:
                    runner.run(produce())
                    result_queue.put(ExecNode._THREAD_QUEUE_SENTINEL)
                except BaseException as e:
                    result_queue.put(e)
                finally:
                    # the clients are bound to this loop, so they have to be closed while it is open
                    runner.run(thread_runtime.close_clients())
                    telemetry.exit_context(hooks_token)

        def is_final(item: Any) -> bool:
            return item is ExecNode._THREAD_QUEUE_SENTINEL or isinstance(item, BaseException)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        item: Any = None
        try:
            while not is_final(item := result_queue.get()):
                yield item
            if isinstance(item, BaseException):
                raise item
        finally:
            # item isn't final if the consumer stopped early: it closed this generator at the yield, or an exception
            # (eg, KeyboardInterrupt) interrupted get()
            if not is_final(item):
                if producer_task is not None:
                    # don't wait for the producer's current batch (eg, slow UDF calls); cancelling also runs the nodes'
                    # cleanup right away
                    try:
                        producer_task.get_loop().call_soon_threadsafe(producer_task.cancel)
                    except RuntimeError:
                        pass  # the producer finished and closed its loop; its final item is in the queue
                # a cancellation can't interrupt a blocking put(), so drain the queue up to the producer's final item
                while not is_final(item):
                    item = result_queue.get()
            thread.join(timeout=30)

    def __enter__(self) -> Self:
        if self.ctx.show_progress:
            self.ctx.start_progress()
        self._open_aux()
        return self

    def _open_aux(self) -> None:
        """Call _open() bottom-up"""
        if self.input is not None:
            self.input._open_aux()
        self._open()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        # Ensure progress stops on exit (including empty results, errors, interrupts)
        get_runtime().stop_progress()
        self._close_aux()

    def _close_aux(self) -> None:
        """Call _close() top-down"""
        self._close()
        if self.input is not None:
            self.input._close_aux()

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    T = TypeVar('T', bound='ExecNode')

    def get_node(self, node_class: type[T]) -> T | None:
        if isinstance(self, node_class):
            return self
        if self.input is not None:
            return self.input.get_node(node_class)
        return None

    def set_limit(self, limit: exprs.Expr) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_limit(limit)

    def set_offset(self, offset: exprs.Expr) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_offset(offset)

    def _resolve_positive_int(self, e: exprs.Expr, role: str) -> int:
        """Resolve Literal or Variable to a positive int value."""
        if isinstance(e, exprs.Literal):
            val = e.val
        elif isinstance(e, exprs.Variable):
            val = self.bound_args[e._bind_name]
        else:
            raise excs.RequestError(
                excs.ErrorCode.UNSUPPORTED_OPERATION, f'{role}: unsupported expression for {role!r}: {e}'
            )
        if val < 0:
            raise excs.RequestError(excs.ErrorCode.INVALID_ARGUMENT, f'{role!r} parameter must be >= 0')
        return val

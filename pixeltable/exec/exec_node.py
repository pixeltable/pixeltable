from __future__ import annotations

import abc
import asyncio
import logging
import queue
import threading
from types import TracebackType
from typing import AsyncIterator, Iterable, Iterator, TypeVar

from typing_extensions import Self

from pixeltable import exprs
from pixeltable.runtime import get_runtime

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext

_logger = logging.getLogger('pixeltable')


class ExecNode(abc.ABC):
    """
    Base class of all execution nodes
    """

    output_exprs: Iterable[exprs.Expr]
    row_builder: exprs.RowBuilder
    input: ExecNode | None
    flushed_img_slots: list[int]  # idxs of image slots of our output_exprs dependencies
    ctx: ExecContext | None

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

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        pass

    def __iter__(self) -> Iterator[DataRowBatch]:
        # Check if we're inside a running event loop that isn't patched by nest_asyncio
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None:
            yield from self._thread_iter()
        else:
            loop = get_runtime().event_loop
            aiter = self.__aiter__()
            try:
                while True:
                    batch: DataRowBatch = loop.run_until_complete(aiter.__anext__())
                    yield batch
            except StopAsyncIteration:
                pass

    _SENTINEL = object()

    def _thread_iter(self) -> Iterator[DataRowBatch]:
        # maxsize=2: we want a minimal amount of buffering
        result_queue: queue.Queue = queue.Queue(maxsize=2)
        caller_runtime = get_runtime()

        def run() -> None:
            thread_runtime = get_runtime()
            # the execution needs to happen in the same db context as the caller, but on a new event loop
            thread_runtime.copy_db_context(caller_runtime)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:

                async def produce() -> None:
                    async for batch in aiter(self):
                        result_queue.put(batch)

                loop.run_until_complete(produce())
                result_queue.put(ExecNode._SENTINEL)
            except BaseException as e:
                result_queue.put(e)
            finally:
                loop.close()
                thread_runtime.conn = None
                thread_runtime.session = None
                thread_runtime.isolation_level = None

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            while True:
                item = result_queue.get()
                if item is ExecNode._SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
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

    def set_limit(self, limit: int) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_limit(limit)

    def set_offset(self, offset: int) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_offset(offset)

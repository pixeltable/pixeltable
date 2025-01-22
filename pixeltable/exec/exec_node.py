from __future__ import annotations

import abc
import asyncio
import logging
import sys
from typing import Iterable, Iterator, Optional, TypeVar, AsyncIterator

import pixeltable.exprs as exprs
from .data_row_batch import DataRowBatch
from .exec_context import ExecContext

_logger = logging.getLogger('pixeltable')

class ExecNode(abc.ABC):
    """Base class of all execution nodes"""
    output_exprs: Iterable[exprs.Expr]
    row_builder: exprs.RowBuilder
    input: Optional[ExecNode]
    flushed_img_slots: list[int]  # idxs of image slots of our output_exprs dependencies
    stored_img_cols: list[exprs.ColumnSlotIdx]
    ctx: Optional[ExecContext]

    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr],
            input_exprs: Iterable[exprs.Expr], input: Optional[ExecNode] = None):
        self.output_exprs = output_exprs
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies
            if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.stored_img_cols = []
        self.ctx = None  # all nodes of a tree share the same context

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def set_stored_img_cols(self, stored_img_cols: list[exprs.ColumnSlotIdx]) -> None:
        self.stored_img_cols = stored_img_cols
        # propagate batch size to the source
        if self.input is not None:
            self.input.set_stored_img_cols(stored_img_cols)

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        pass

    def __iter__(self) -> Iterator[DataRowBatch]:
        try:
            # check if we are already in an event loop (eg, Jupyter's); if so, patch it to allow nested event loops
            _ = asyncio.get_event_loop()
            import nest_asyncio  # type: ignore
            nest_asyncio.apply()
        except RuntimeError:
            pass

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if 'pytest' in sys.modules:
            loop.set_debug(True)

        aiter = self.__aiter__()
        try:
            while True:
                batch: DataRowBatch = loop.run_until_complete(aiter.__anext__())
                yield batch
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    def open(self) -> None:
        """Bottom-up initialization of nodes for execution. Must be called before __next__."""
        if self.input is not None:
            self.input.open()
        self._open()

    def close(self) -> None:
        """Frees node resources top-down after execution. Must be called after final __next__."""
        self._close()
        if self.input is not None:
            self.input.close()

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    T = TypeVar('T', bound='ExecNode')

    def get_node(self, node_class: type[T]) -> Optional[T]:
        if isinstance(self, node_class):
            return self
        if self.input is not None:
            return self.input.get_node(node_class)
        return None

    def set_limit(self, limit: int) -> None:
        """Default implementation propagates to input"""
        if self.input is not None:
            self.input.set_limit(limit)

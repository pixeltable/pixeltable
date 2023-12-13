from __future__ import annotations
from typing import Iterable, Optional, List
import abc

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext
import pixeltable.exprs as exprs

class ExecNode(abc.ABC):
    """Base class of all execution nodes"""
    def __init__(
            self, row_builder: exprs.RowBuilder, output_exprs: Iterable[exprs.Expr],
            input_exprs: Iterable[exprs.Expr], input: Optional[ExecNode] = None):
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies
            if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.stored_img_cols: List[exprs.ColumnSlotIdx] = []
        self.ctx: Optional[ExecContext] = None  # all nodes of a tree share the same context

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def set_stored_img_cols(self, stored_img_cols: List[exprs.ColumnSlotIdx]) -> None:
        self.stored_img_cols = stored_img_cols
        # propagate batch size to the source
        if self.input is not None:
            self.input.set_stored_img_cols(stored_img_cols)

    def __iter__(self):
        return self

    @abc.abstractmethod
    def __next__(self) -> DataRowBatch:
        pass

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


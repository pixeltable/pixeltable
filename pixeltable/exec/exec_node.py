from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Iterable, Iterator, Optional, TypeVar

import pixeltable.exprs as exprs

from .data_row_batch import DataRowBatch
from .exec_context import ExecContext


class ExecNode(abc.ABC):
    """Base class of all execution nodes"""

    output_exprs: Iterable[exprs.Expr]
    row_builder: exprs.RowBuilder
    input: Optional[ExecNode]
    flushed_img_slots: list[int]  # idxs of image slots of our output_exprs dependencies
    stored_img_cols: list[exprs.ColumnSlotIdx]
    ctx: Optional[ExecContext]
    __iter: Optional[Iterator[DataRowBatch]]

    def __init__(
        self,
        row_builder: exprs.RowBuilder,
        output_exprs: Iterable[exprs.Expr],
        input_exprs: Iterable[exprs.Expr],
        input: Optional[ExecNode] = None,
    ):
        self.output_exprs = output_exprs
        self.row_builder = row_builder
        self.input = input
        # we flush all image slots that aren't part of our output but are needed to create our output
        output_slot_idxs = {e.slot_idx for e in output_exprs}
        output_dependencies = row_builder.get_dependencies(output_exprs, exclude=input_exprs)
        self.flushed_img_slots = [
            e.slot_idx for e in output_dependencies if e.col_type.is_image_type() and e.slot_idx not in output_slot_idxs
        ]
        self.stored_img_cols = []
        self.ctx = None  # all nodes of a tree share the same context
        self.__iter = None

    def set_ctx(self, ctx: ExecContext) -> None:
        self.ctx = ctx
        if self.input is not None:
            self.input.set_ctx(ctx)

    def set_stored_img_cols(self, stored_img_cols: list[exprs.ColumnSlotIdx]) -> None:
        self.stored_img_cols = stored_img_cols
        # propagate batch size to the source
        if self.input is not None:
            self.input.set_stored_img_cols(stored_img_cols)

    # TODO: make this an abstractmethod when __next__() is removed
    def __iter__(self) -> Iterator[DataRowBatch]:
        return self

    # TODO: remove this and switch every subclass over to implementing __iter__
    def __next__(self) -> DataRowBatch:
        if self.__iter is None:
            self.__iter = iter(self)
        return next(self.__iter)

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

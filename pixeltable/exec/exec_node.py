from __future__ import annotations

import abc
import logging
from types import TracebackType
from typing import AsyncIterator, Iterable, Iterator, TypeVar

from typing_extensions import Self

from pixeltable import exprs
from pixeltable.env import Env

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
        loop = Env.get().event_loop
        aiter = self.__aiter__()
        try:
            while True:
                batch: DataRowBatch = loop.run_until_complete(aiter.__anext__())
                yield batch
        except StopAsyncIteration:
            pass
        # TODO:
        #  - we seem to have some tasks that aren't accounted for by ExprEvalNode and don't get cancelled by the time
        #    we end up here
        # - however, blindly cancelling all pending tasks doesn't work when running in a jupyter environment, which
        #   creates tasks on its own

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

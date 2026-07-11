from __future__ import annotations

from typing import AsyncIterator

from pixeltable import exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode


class FilterNode(ExecNode):
    """
    Drops rows that don't pass a pre-evaluated boolean predicate (needed for non-SQL where() clause predicates).

    The predicate slot must already be populated by an upstream ExprEvalNode.
    Applies offset/limit after filtering (intercepting set_limit/set_offset so they
    are not propagated to SQL).
    """

    predicate_slot_idx: int
    limit: exprs.Expr | None
    offset: exprs.Expr | None

    def __init__(self, row_builder: exprs.RowBuilder, predicate: exprs.Expr, input: ExecNode):
        super().__init__(row_builder, [], [], input)
        self.predicate_slot_idx = predicate.slot_idx
        self.limit = None
        self.offset = None

    def set_limit(self, limit: exprs.Expr) -> None:
        self.limit = limit

    def set_offset(self, offset: exprs.Expr) -> None:
        self.offset = offset

    def init_bindings(self) -> None:
        self.bind_sources = [e for e in (self.limit, self.offset) if e is not None]
        super().init_bindings()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        limit = self._resolve_positive_int(self.limit, 'limit') if self.limit is not None else None
        offset = self._resolve_positive_int(self.offset, 'offset') if self.offset is not None else None
        if limit == 0:
            return
        num_passed = 0  # rows that passed the predicate (before offset/limit)
        limit_reached = False

        async for batch in self.input:
            if limit_reached:
                break
            output_batch = DataRowBatch(self.row_builder)
            for row in batch:
                # a predicate whose evaluation failed (recorded under ignore_errors) counts as not passing
                if row.has_exc(self.predicate_slot_idx) or not row[self.predicate_slot_idx]:
                    continue
                num_passed += 1
                if offset is not None and num_passed <= offset:
                    continue
                output_batch.add_row(row)
                if limit is not None and num_passed - (offset or 0) >= limit:
                    limit_reached = True
                    break
            if len(output_batch) > 0:
                yield output_batch

from __future__ import annotations

import logging
from typing import AsyncIterator

from pixeltable import exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class FilterNode(ExecNode):
    """
    Drops rows that don't pass a pre-evaluated boolean predicate (needed for non-SQL where() clause predicates).

    The predicate slot must already be populated by an upstream ExprEvalNode.
    Applies offset/limit after filtering (intercepting set_limit/set_offset so they
    are not propagated to SQL).
    """

    predicate_slot_idx: int
    limit: int | None
    offset: int | None

    def __init__(self, row_builder: exprs.RowBuilder, predicate: exprs.Expr, input: ExecNode):
        super().__init__(row_builder, [], [], input)
        self.predicate_slot_idx = predicate.slot_idx
        self.limit = None
        self.offset = None

    def set_limit(self, limit: int) -> None:
        assert limit > 0
        self.limit = limit

    def set_offset(self, offset: int) -> None:
        self.offset = offset

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        num_passed = 0  # rows that passed the predicate (before offset/limit)
        limit_reached = False

        async for batch in self.input:
            if limit_reached:
                break
            output_batch = DataRowBatch(self.row_builder)
            for row in batch:
                if not row[self.predicate_slot_idx]:
                    continue
                num_passed += 1
                if self.offset is not None and num_passed <= self.offset:
                    continue
                output_batch.add_row(row)
                if self.limit is not None and num_passed - (self.offset or 0) >= self.limit:
                    limit_reached = True
                    break
            if len(output_batch) > 0:
                yield output_batch

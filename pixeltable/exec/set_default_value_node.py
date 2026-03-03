from __future__ import annotations

import logging
from typing import AsyncIterator

from pixeltable import catalog

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class SetDefaultValueNode(ExecNode):
    """
    Fills default values for specified columns (only when slot is missing value).
    """

    output_col_info: dict[catalog.Column, int]  # col -> slot_idx

    def __init__(self, input: ExecNode, columns_with_defaults: list[catalog.Column] | None = None):
        super().__init__(input.row_builder, list(input.output_exprs), [], input)

        if columns_with_defaults is not None:
            self.output_col_info = {
                col: slot_idx
                for col in columns_with_defaults
                if (slot_idx := input.row_builder.table_columns.get(col)) is not None
                and col.has_default_value
                and col.default_value_expr is not None
            }
        else:
            self.output_col_info = {
                col: slot_idx
                for col, slot_idx in input.row_builder.table_columns.items()
                if slot_idx is not None and col.is_stored and col.has_default_value
            }

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                for col, slot_idx in self.output_col_info.items():
                    if not row.has_val[slot_idx]:
                        assert col.default_value_expr is not None
                        row[slot_idx] = col.default_value_expr.val
            yield batch

from __future__ import annotations
from typing import Iterable, Optional

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.exprs as exprs

class ValidateDataNode(ExecNode):
    """Base class of all execution nodes"""
    def __init__(self,
            row_builder: exprs.RowBuilder,
            input: Optional[ExecNode],
            ignore_errors: bool):
        super().__init__(row_builder, [], [], input)
        self.row_builder = row_builder
        self.ignore_errors = ignore_errors
        self.input = input

    def __next__(self) -> DataRowBatch:
        """Return the next batch of rows"""
        assert self.input is not None
        row_batch = next(self.input)
        for row_idx in range(len(row_batch)):
            self.row_builder.validate(row_batch[row_idx], ignore_errors=self.ignore_errors)
        return row_batch

from __future__ import annotations
from typing import Iterable, Optional

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.exprs as exprs
import pixeltable.exceptions as excs


class MediaValidationNode(ExecNode):
    """Validation of selected media slots
    Records exceptions in the rows of the input batch
    """
    def __init__(
            self, row_builder: exprs.RowBuilder, media_slots: Iterable[exprs.ColumnSlotIdx],
            input: Optional[ExecNode]):
        super().__init__(row_builder, [], [], input)
        self.row_builder = row_builder
        self.input = input
        for col in [c.col for c in media_slots]:
            assert col.col_type.is_media_type()
        self.media_slots = media_slots

    def __next__(self) -> DataRowBatch:
        assert self.input is not None
        row_batch = next(self.input)
        for row in row_batch:
            for slot_idx, col in [(c.slot_idx, c.col) for c in self.media_slots]:
                if row.has_exc(slot_idx):
                    continue
                assert row.has_val[slot_idx]
                path = row.file_paths[slot_idx]
                if path is None:
                    continue

                try:
                    col.col_type.validate_media(path)
                except excs.Error as exc:
                    self.row_builder.set_exc(row, slot_idx, exc)
                    if not self.ctx.ignore_errors:
                        raise exc

        return row_batch

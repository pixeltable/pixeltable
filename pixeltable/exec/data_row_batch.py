from __future__ import annotations

import logging
from typing import Iterator, Optional

from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


class DataRowBatch:
    """Set of DataRows, indexed by rowid.

    Contains the metadata needed to initialize DataRows.
    """

    row_builder: exprs.RowBuilder
    rows: list[exprs.DataRow]

    def __init__(self, row_builder: exprs.RowBuilder, rows: Optional[list[exprs.DataRow]] = None):
        """
        Requires either num_rows or rows to be specified, but not both.
        """
        self.row_builder = row_builder
        self.rows = [] if rows is None else rows

    def add_row(self, row: Optional[exprs.DataRow]) -> exprs.DataRow:
        if row is None:
            row = self.row_builder.make_row()
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        return self.rows.pop()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> exprs.DataRow:
        return self.rows[index]

    def flush_imgs(
        self, idx_range: Optional[slice], stored_img_info: list[exprs.ColumnSlotIdx], flushed_img_slots: list[int]
    ) -> None:
        """Flushes images in the given range of rows."""
        if len(stored_img_info) == 0 and len(flushed_img_slots) == 0:
            return

        if idx_range is None:
            idx_range = slice(0, len(self.rows))
        for row in self.rows[idx_range]:
            for info in stored_img_info:
                row.flush_img(info.slot_idx, info.col)
            for slot_idx in flushed_img_slots:
                row.flush_img(slot_idx)

    def __iter__(self) -> Iterator[exprs.DataRow]:
        return iter(self.rows)

from __future__ import annotations

import logging
from typing import Iterator

from pixeltable import exprs

_logger = logging.getLogger('pixeltable')


class DataRowBatch:
    """Set of DataRows, indexed by rowid.

    Contains the metadata needed to initialize DataRows.

    Requires either num_rows or rows to be specified, but not both.
    """

    row_builder: exprs.RowBuilder
    rows: list[exprs.DataRow]

    def __init__(self, row_builder: exprs.RowBuilder, rows: list[exprs.DataRow] | None = None):
        self.row_builder = row_builder
        self.rows = [] if rows is None else rows

    def add_row(self, row: exprs.DataRow | None) -> exprs.DataRow:
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

    def __iter__(self) -> Iterator[exprs.DataRow]:
        return iter(self.rows)

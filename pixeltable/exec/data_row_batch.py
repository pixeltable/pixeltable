from __future__ import annotations

import logging
from typing import Iterator, Optional

from pixeltable import catalog, exprs
from pixeltable.utils.media_store import MediaStore

_logger = logging.getLogger('pixeltable')


class DataRowBatch:
    """Set of DataRows, indexed by rowid.

    Contains the metadata needed to initialize DataRows.
    """

    tbl: Optional[catalog.TableVersionHandle]
    row_builder: exprs.RowBuilder
    img_slot_idxs: list[int]
    media_slot_idxs: list[int]  # non-image media slots
    array_slot_idxs: list[int]
    rows: list[exprs.DataRow]

    def __init__(
        self,
        tbl: Optional[catalog.TableVersionHandle],
        row_builder: exprs.RowBuilder,
        num_rows: Optional[int] = None,
        rows: Optional[list[exprs.DataRow]] = None,
    ):
        """
        Requires either num_rows or rows to be specified, but not both.
        """
        assert num_rows is None or rows is None
        self.tbl = tbl
        self.row_builder = row_builder
        self.img_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_image_type()]
        # non-image media slots
        self.media_slot_idxs = [
            e.slot_idx
            for e in row_builder.unique_exprs
            if e.col_type.is_media_type() and not e.col_type.is_image_type()
        ]
        self.array_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_array_type()]
        if rows is not None:
            self.rows = rows
        else:
            if num_rows is None:
                num_rows = 0
            self.rows = [
                exprs.DataRow(
                    row_builder.num_materialized, self.img_slot_idxs, self.media_slot_idxs, self.array_slot_idxs
                )
                for _ in range(num_rows)
            ]

    def add_row(self, row: Optional[exprs.DataRow] = None) -> exprs.DataRow:
        if row is None:
            row = exprs.DataRow(
                self.row_builder.num_materialized, self.img_slot_idxs, self.media_slot_idxs, self.array_slot_idxs
            )
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        return self.rows.pop()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> exprs.DataRow:
        return self.rows[index]

    def flush_imgs(
        self,
        idx_range: Optional[slice] = None,
        stored_img_info: Optional[list[exprs.ColumnSlotIdx]] = None,
        flushed_slot_idxs: Optional[list[int]] = None,
    ) -> None:
        """Flushes images in the given range of rows."""
        assert self.tbl is not None
        if stored_img_info is None:
            stored_img_info = []
        if flushed_slot_idxs is None:
            flushed_slot_idxs = []
        if len(stored_img_info) == 0 and len(flushed_slot_idxs) == 0:
            return
        if idx_range is None:
            idx_range = slice(0, len(self.rows))
        for row in self.rows[idx_range]:
            for info in stored_img_info:
                filepath = str(MediaStore.prepare_media_path(self.tbl.id, info.col.id, self.tbl.get().version))
                row.flush_img(info.slot_idx, filepath)
            for slot_idx in flushed_slot_idxs:
                row.flush_img(slot_idx)

    def __iter__(self) -> Iterator[exprs.DataRow]:
        return iter(self.rows)

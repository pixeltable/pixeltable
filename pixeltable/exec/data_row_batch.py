from __future__ import annotations
from typing import List, Iterator, Optional
import logging

import pixeltable.exprs as exprs
import pixeltable.catalog as catalog
from pixeltable.utils.media_store import MediaStore


_logger = logging.getLogger('pixeltable')

class DataRowBatch:
    """Set of DataRows, indexed by rowid.

    Contains the metadata needed to initialize DataRows.
    """
    def __init__(self, tbl: catalog.TableVersion, row_builder: exprs.RowBuilder, len: int = 0):
        self.tbl_id = tbl.id
        self.tbl_version = tbl.version
        self.row_builder = row_builder
        self.img_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_image_type()]
        # non-image media slots
        self.media_slot_idxs = [
            e.slot_idx for e in row_builder.unique_exprs
            if e.col_type.is_media_type() and not e.col_type.is_image_type()
        ]
        self.array_slot_idxs = [e.slot_idx for e in row_builder.unique_exprs if e.col_type.is_array_type()]
        self.rows = [
            exprs.DataRow(row_builder.num_materialized, self.img_slot_idxs, self.media_slot_idxs, self.array_slot_idxs)
            for _ in range(len)
        ]

    def add_row(self, row: Optional[exprs.DataRow] = None) -> exprs.DataRow:
        if row is None:
            row = exprs.DataRow(
                self.row_builder.num_materialized, self.img_slot_idxs, self.media_slot_idxs, self.array_slot_idxs)
        self.rows.append(row)
        return row

    def pop_row(self) -> exprs.DataRow:
        return self.rows.pop()

    def set_row_ids(self, row_ids: List[int]) -> None:
        """Sets pks for rows in batch"""
        assert len(row_ids) == len(self.rows)
        for row, row_id in zip(self.rows, row_ids):
            row.set_pk((row_id, self.tbl_version))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: object) -> exprs.DataRow:
        return self.rows[index]

    def flush_imgs(
            self, idx_range: Optional[slice] = None, stored_img_info: Optional[List[exprs.ColumnSlotIdx]] = None,
            flushed_slot_idxs: Optional[List[int]] = None
    ) -> None:
        """Flushes images in the given range of rows."""
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
                filepath = str(MediaStore.prepare_media_path(self.tbl_id, info.col.id, self.tbl_version))
                row.flush_img(info.slot_idx, filepath)
            for slot_idx in flushed_slot_idxs:
                row.flush_img(slot_idx)
        #_logger.debug(
            #f'flushed images in range {idx_range}: slot_idxs={flushed_slot_idxs} stored_img_info={stored_img_info}')

    def __iter__(self) -> Iterator[exprs.DataRow]:
        return DataRowBatchIterator(self)


class DataRowBatchIterator:
    """
    Iterator over a DataRowBatch.
    """
    def __init__(self, batch: DataRowBatch):
        self.row_batch = batch
        self.index = 0

    def __next__(self) -> exprs.DataRow:
        if self.index >= len(self.row_batch.rows):
            raise StopIteration
        row = self.row_batch.rows[self.index]
        self.index += 1
        return row


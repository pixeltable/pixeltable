from typing import List, Dict, Any, Optional
import urllib
import logging
import os

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
import pixeltable.env as env
from pixeltable.utils.media_store import MediaStore


_logger = logging.getLogger('pixeltable')

class InMemoryDataNode(ExecNode):
    """Outputs in-memory data as a row batch of a particular table"""
    def __init__(
            self, tbl: catalog.TableVersionPath, rows: List[Dict[str, Any]],
            row_builder: exprs.RowBuilder, start_row_id: int,
    ):
        super().__init__(row_builder, [], [], None)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.input_rows = rows
        self.start_row_id = start_row_id
        self.has_returned_data = False
        self.output_rows: Optional[DataRowBatch] = None

    def _open(self) -> None:
        """Create row batch and populate with self.input_rows"""
        column_info = {info.col.name: info for info in self.row_builder.output_slot_idxs()}
        # stored columns that are not computed
        inserted_column_names = set([
            info.col.name for info in self.row_builder.output_slot_idxs()
            if info.col.is_stored and not info.col.is_computed
        ])

        self.output_rows = DataRowBatch(self.tbl, self.row_builder, len(self.input_rows))
        for row_idx, input_row in enumerate(self.input_rows):
            # populate the output row with the values provided in the input row
            for col_name, val in input_row.items():
                col_info = column_info.get(col_name)
                assert col_info is not None

                if col_info.col.col_type.is_image_type() and isinstance(val, bytes):
                    # this is a literal image, ie, a sequence of bytes; we save this as a media file and store the path
                    path = str(MediaStore.prepare_media_path(self.tbl.id, col_info.col.id, self.tbl.version))
                    open(path, 'wb').write(val)
                    val = path
                self.output_rows[row_idx][col_info.slot_idx] = val

            # set the remaining stored non-computed columns to null
            null_col_names = inserted_column_names - set(input_row.keys())
            for col_name in null_col_names:
                col_info = column_info.get(col_name)
                assert col_info is not None
                self.output_rows[row_idx][col_info.slot_idx] = None

        self.output_rows.set_row_ids([self.start_row_id + i for i in range(len(self.output_rows))])
        self.ctx.num_rows = len(self.output_rows)

    def __next__(self) -> DataRowBatch:
        if self.has_returned_data:
            raise StopIteration
        self.has_returned_data = True
        _logger.debug(f'InMemoryDataNode: created row batch with {len(self.output_rows)} output_rows')
        return self.output_rows


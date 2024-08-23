import logging
from typing import Any, Optional

import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
from pixeltable.utils.media_store import MediaStore
from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')

class InMemoryDataNode(ExecNode):
    """
    Outputs in-memory data as a DataRowBatch of a particular table.

    Populates slots of all non-computed columns (ie, output ColumnRefs)
    - with the values provided in the input rows
    - if an input row doesn't provide a value, sets the slot to the column default
    """
    def __init__(
            self, tbl: catalog.TableVersion, rows: list[dict[str, Any]],
            row_builder: exprs.RowBuilder, start_row_id: int,
    ):
        # we materialize all output slots
        output_exprs = [e for e in row_builder.get_output_exprs() if isinstance(e, exprs.ColumnRef)]
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.input_rows = rows
        self.start_row_id = start_row_id
        self.has_returned_data = False
        self.output_rows: Optional[DataRowBatch] = None

    def _open(self) -> None:
        """Create row batch and populate with self.input_rows"""
        user_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs if col_ref.col.name is not None
        }
        output_cols_by_idx = {
            col_ref.slot_idx: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs
        }
        output_slot_idxs = {e.slot_idx for e in self.output_exprs}

        self.output_rows = DataRowBatch(self.tbl, self.row_builder, len(self.input_rows))
        for row_idx, input_row in enumerate(self.input_rows):
            # populate the output row with the values provided in the input row
            input_slot_idxs: set[int] = set()
            for col_name, val in input_row.items():
                col_info = user_cols_by_name.get(col_name)
                assert col_info is not None

                if col_info.col.col_type.is_image_type() and isinstance(val, bytes):
                    # this is a literal image, ie, a sequence of bytes; we save this as a media file and store the path
                    path = str(MediaStore.prepare_media_path(self.tbl.id, col_info.col.id, self.tbl.version))
                    open(path, 'wb').write(val)
                    val = path
                self.output_rows[row_idx][col_info.slot_idx] = val
                input_slot_idxs.add(col_info.slot_idx)

            # set the remaining output slots to their default values (presently None)
            missing_slot_idxs =  output_slot_idxs - input_slot_idxs
            for slot_idx in missing_slot_idxs:
                col_info = output_cols_by_idx.get(slot_idx)
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


class MapperDataNode(ExecNode):
    """
    Outputs in-memory data as a DataRowBatch of a particular table.

    Populates slots of all non-computed columns (ie, output ColumnRefs)
    - with the values provided in the input rows
    - if an input row doesn't provide a value, sets the slot to the column default
    """
    def __init__(
        self,
        tbl: catalog.TableVersion,
        row_builder: exprs.RowBuilder,
        input_col_idxs: list[str, int],
        input: ExecNode,
    ):
        output_exprs = [e for e in row_builder.get_output_exprs() if isinstance(e, exprs.ColumnRef)]
        super().__init__(row_builder, output_exprs, [], input=input)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.next_rowid = tbl.next_rowid
        self.slot_idx_map: dict[int, int] = {}
        self.missing_slot_idxs: set[int] = set()
        assert all(
            col_name in tbl.cols_by_name and not tbl.cols_by_name[col_name].is_computed
            for col_name in input_col_idxs
        )
        for e in output_exprs:
            col_name = e.col.name
            if col_name is not None and col_name in input_col_idxs:
                input_col_idx = input_col_idxs[col_name]
                self.slot_idx_map[input_col_idx] = e.slot_idx
            else:
                self.missing_slot_idxs.add(e.slot_idx)

    def __next__(self) -> DataRowBatch:
        input_rows = next(self.input)
        output_rows = DataRowBatch(self.tbl, self.row_builder, len(input_rows))
        output_rows.set_row_ids([self.next_rowid + i for i in range(len(input_rows))])
        for row_idx, input_row in enumerate(input_rows):
            for col_idx, val in enumerate(input_row):
                assert col_idx in self.slot_idx_map
                output_rows[row_idx][self.slot_idx_map[col_idx]] = val
            for slot_idx in self.missing_slot_idxs:
                output_rows[row_idx][slot_idx] = None
        self.next_rowid += len(input_rows)
        return output_rows

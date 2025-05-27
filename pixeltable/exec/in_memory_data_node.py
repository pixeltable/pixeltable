import logging
from typing import Any, AsyncIterator, Optional

from pixeltable import catalog, exprs
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

    tbl: catalog.TableVersionHandle

    input_rows: list[dict[str, Any]]
    start_row_id: int
    output_rows: Optional[DataRowBatch]

    # output_exprs is declared in the superclass, but we redeclare it here with a more specific type
    output_exprs: list[exprs.ColumnRef]

    def __init__(
        self,
        tbl: catalog.TableVersionHandle,
        rows: list[dict[str, Any]],
        row_builder: exprs.RowBuilder,
        start_row_id: int,
    ):
        # we materialize the input slots
        output_exprs = list(row_builder.input_exprs)
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.get().is_insertable
        self.tbl = tbl
        self.input_rows = rows
        self.start_row_id = start_row_id
        self.output_rows = None

    def _open(self) -> None:
        """Create row batch and populate with self.input_rows"""
        user_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs
            if col_ref.col.name is not None
        }
        output_cols_by_idx = {
            col_ref.slot_idx: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx) for col_ref in self.output_exprs
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
                    path = str(MediaStore.prepare_media_path(self.tbl.id, col_info.col.id, self.tbl.get().version))
                    with open(path, 'wb') as fp:
                        fp.write(val)
                    self.output_rows[row_idx][col_info.slot_idx] = path
                else:
                    self.output_rows[row_idx][col_info.slot_idx] = val

                input_slot_idxs.add(col_info.slot_idx)

            # set the remaining output slots to their default values (presently None)
            missing_slot_idxs = output_slot_idxs - input_slot_idxs
            for slot_idx in missing_slot_idxs:
                col_info = output_cols_by_idx.get(slot_idx)
                assert col_info is not None
                self.output_rows[row_idx][col_info.slot_idx] = None

        self.ctx.num_rows = len(self.output_rows)

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        _logger.debug(f'InMemoryDataNode: created row batch with {len(self.output_rows)} output_rows')
        yield self.output_rows

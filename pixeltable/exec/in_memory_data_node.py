import logging
from typing import Any, AsyncIterator

from pixeltable import catalog, exprs
from pixeltable.utils.local_store import TempStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger(__name__)


class InMemoryDataNode(ExecNode):
    """
    Outputs in-memory data as a DataRowBatch of a particular table.

    Populates slots of all non-computed columns (ie, output ColumnRefs)
    - with the values provided in the input rows
    - if an input row doesn't provide a value, sets the slot to the column default
    """

    tbl: catalog.TableVersionHandle

    input_rows: list[dict[str, Any]]
    set_pk: bool  # if True, assign synthetic pks (input row position); needed for iterator expansion
    output_batch: DataRowBatch | None

    # output_exprs is declared in the superclass, but we redeclare it here with a more specific type
    output_exprs: list[exprs.ColumnRef]

    def __init__(
        self,
        tbl: catalog.TableVersionHandle,
        rows: list[dict[str, Any]],
        row_builder: exprs.RowBuilder,
        set_pk: bool = False,
    ):
        # we materialize the input slots
        output_exprs = list(row_builder.input_exprs)
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.get().is_insertable
        self.tbl = tbl
        self.input_rows = rows
        self.set_pk = set_pk
        self.output_batch = None

    def _open(self) -> None:
        """Create row batch and populate with self.input_rows"""
        # input rows can only provide values for this table's columns
        user_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs
            if col_ref.col.name is not None and col_ref.col.get_tbl().id == self.tbl.id
        }
        output_cols_by_idx = {
            col_ref.slot_idx: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx) for col_ref in self.output_exprs
        }
        output_slot_idxs = {e.slot_idx for e in self.output_exprs}

        self.output_batch = DataRowBatch(self.row_builder)
        for row_idx, input_row in enumerate(self.input_rows):
            output_row = self.row_builder.make_row()
            if self.set_pk:
                # (rowid, v_min); the rowid stands in for a store-assigned one, the version is irrelevant here
                output_row.set_pk((row_idx, 0))
            # populate the output row with the values provided in the input row
            input_slot_idxs: set[int] = set()
            for col_name, val in input_row.items():
                col_info = user_cols_by_name.get(col_name)
                assert col_info is not None
                col = col_info.col
                if col.col_type.is_image_type() and isinstance(val, bytes):
                    # this is a literal media file, ie, a sequence of bytes; save it as a binary file and store the path
                    filepath, _ = TempStore.save_media_object(
                        val, col.tbl_handle.id, col.id, col.get_tbl().version, format=None
                    )
                    output_row[col_info.slot_idx] = str(filepath)
                else:
                    output_row[col_info.slot_idx] = val

                input_slot_idxs.add(col_info.slot_idx)

            # set the remaining output slots to their default values (presently None)
            missing_slot_idxs = output_slot_idxs - input_slot_idxs
            for slot_idx in missing_slot_idxs:
                col_info = output_cols_by_idx.get(slot_idx)
                assert col_info is not None
                output_row[col_info.slot_idx] = None
            self.output_batch.add_row(output_row)

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        _logger.debug(f'InMemoryDataNode: created row batch with {len(self.output_batch)} rows')
        yield self.output_batch

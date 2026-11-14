import logging
from typing import Any, AsyncGenerator, Iterable, Iterator

from pixeltable import catalog, exprs
from pixeltable.utils.local_store import TempStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger(__name__)


class RowDataNode(ExecNode):
    """
    Outputs a stream of row dicts as DataRowBatches of a particular table, ctx.batch_size rows at a time (or all in one
    batch if ctx.batch_size is 0).

    Populates slots of all non-computed columns (ie, output ColumnRefs)
    - with the values provided in the input rows
    - if an input row doesn't provide a value, sets the slot to the column default
    """

    tbl: catalog.TableVersionHandle

    input_rows: Iterator[dict[str, Any]]
    set_pk: bool  # if True, assign synthetic pks (input row position); needed for iterator expansion
    _is_consumed: bool
    _is_closed: bool

    # output_exprs is declared in the superclass, but we redeclare it here with a more specific type
    output_exprs: list[exprs.ColumnRef]

    def __init__(
        self,
        tbl: catalog.TableVersionHandle,
        rows: Iterable[dict[str, Any]],
        row_builder: exprs.RowBuilder,
        set_pk: bool = False,
    ):
        # we materialize the input slots
        output_exprs = list(row_builder.input_exprs)
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.get().is_insertable
        self.tbl = tbl
        self.input_rows = iter(rows)
        self.set_pk = set_pk
        self._is_consumed = False
        self._is_closed = False

    def _open(self) -> None:
        assert not self._is_closed, 'RowDataNode is not reusable'

    def _close(self) -> None:
        # Release references to input rows. When images are present, this may free up a significant amount of memory.
        self.input_rows = iter(())
        self._is_closed = True

    async def __aiter__(self) -> AsyncGenerator[DataRowBatch, None]:
        assert not self._is_consumed, 'RowDataNode can only be iterated once'
        self._is_consumed = True
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

        output_batch = DataRowBatch(self.row_builder)
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
            output_batch.add_row(output_row)
            if self.ctx.batch_size > 0 and len(output_batch) >= self.ctx.batch_size:
                _logger.debug(f'RowDataNode: yielding batch of {len(output_batch)} rows')
                yield output_batch
                output_batch = DataRowBatch(self.row_builder)

        if len(output_batch) > 0:
            _logger.debug(f'RowDataNode: yielding batch of {len(output_batch)} rows')
            yield output_batch

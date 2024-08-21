import logging
from typing import Any, Iterable, Iterator, Optional

import pixeltable.catalog as catalog
import pixeltable.exprs as exprs
from pixeltable.utils.media_store import MediaStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class DataNode(ExecNode):
    """
    Outputs batches of explicitly specified data as DataRowBatches of a particular table.

    Populates slots of all non-computed columns (ie, output ColumnRefs)
    - with the values provided in the input rows
    - if an input row doesn't provide a value, sets the slot to the column default
    """

    tbl: catalog.TableVersionPath
    next_row_id: int
    input_row_iter: Iterator[list[dict[str, Any]]]
    total_rows: int
    validate_media: bool
    user_cols_by_name: dict[str, exprs.ColumnSlotIdx]
    output_cols_by_idx: dict[int, exprs.ColumnSlotIdx]
    output_slot_idxs: set[int]

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        input_row_batches: Iterable[list[dict[str, Any]]],
        row_builder: exprs.RowBuilder,
        start_row_id: int,
        total_rows: int = 0,
        validate_media: bool = True
    ):
        # we materialize all output slots
        output_exprs = [e for e in row_builder.get_output_exprs() if isinstance(e, exprs.ColumnRef)]
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.is_insertable()
        self.tbl = tbl
        self.next_row_id = start_row_id
        self.input_row_iter: Iterator[list[dict[str, Any]]] = iter(input_row_batches)
        self.total_rows = total_rows
        self.validate_media = validate_media

        self.user_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs if col_ref.col.name is not None
        }
        self.output_cols_by_idx = {
            col_ref.slot_idx: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs
        }
        self.output_slot_idxs = {e.slot_idx for e in self.output_exprs}

    def _open(self) -> None:
        self.ctx.num_rows = self.total_rows

    def __next__(self) -> DataRowBatch:
        input_rows = next(self.input_row_iter)
        output_rows = DataRowBatch(self.tbl, self.row_builder, len(input_rows))

        for row_idx, input_row in enumerate(input_rows):
            # populate the output row with the values provided in the input row
            input_slot_idxs: set[int] = set()
            for col_name, val in input_row.items():
                col_info = self.user_cols_by_name.get(col_name)
                assert col_info is not None
                if self.validate_media and col_info.col.col_type.is_image_type() and isinstance(val, bytes):
                    # this is a literal image, ie, a sequence of bytes; we save this as a media file and store the path
                    path = str(MediaStore.prepare_media_path(self.tbl.id, col_info.col.id, self.tbl.version))
                    open(path, 'wb').write(val)
                    val = path
                output_rows[row_idx][col_info.slot_idx] = val
                input_slot_idxs.add(col_info.slot_idx)

            # set the remaining output slots to their default values (presently None)
            missing_slot_idxs = self.output_slot_idxs - input_slot_idxs
            for slot_idx in missing_slot_idxs:
                col_info = self.output_cols_by_idx.get(slot_idx)
                assert col_info is not None
                output_rows[row_idx][col_info.slot_idx] = None

        output_rows.set_row_ids([self.next_row_id + i for i in range(len(output_rows))])
        self.next_row_id += len(output_rows)
        _logger.debug(f'DataNode: created row batch with {len(output_rows)} output_rows')
        return output_rows

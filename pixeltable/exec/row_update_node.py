import logging
from typing import Any, AsyncIterator

from pixeltable import catalog, exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


class RowUpdateNode(ExecNode):
    """
    Update individual rows in the input batches, identified by key columns.

    The updates for a row are provided as a dict of column names to new values.
    Populates the slots of the columns present in the update list.
    Assumptions:
    - all update dicts contain the same keys
    - the input node populates DataRow.cell_vals for all primary key columns
    """

    updates: dict[tuple, dict[catalog.Column, Any]]
    is_rowid_key: bool  # if True, key_vals_batch contains rowids rather than primary key values
    col_slot_idxs: dict[catalog.Column, int]
    pk_columns: list[catalog.Column]
    matched_key_vals: set[tuple]

    def __init__(
        self,
        tbl: catalog.TableVersionPath,
        key_vals_batch: list[tuple],
        is_rowid_key: bool,
        col_vals_batch: list[dict[catalog.Column, Any]],
        row_builder: exprs.RowBuilder,
        input: ExecNode,
    ):
        super().__init__(row_builder, [], [], input)
        self.updates = dict(zip(key_vals_batch, col_vals_batch))
        self.is_rowid_key = is_rowid_key
        # determine slot idxs of all columns we need to read or write
        # retrieve ColumnRefs from the RowBuilder (has slot_idx set)
        all_col_slot_idxs = {
            col_ref.col: col_ref.slot_idx
            for col_ref in row_builder.unique_exprs
            if isinstance(col_ref, exprs.ColumnRef)
        }
        # all update target columns should have assigned slot idxs
        assert all(col in all_col_slot_idxs for col in col_vals_batch[0])
        self.col_slot_idxs = {col: all_col_slot_idxs[col] for col in col_vals_batch[0]}
        self.pk_columns = tbl.tbl_version.get().primary_key_columns()
        self.matched_key_vals = set()

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        async for batch in self.input:
            for row in batch:
                key_vals = row.rowid if self.is_rowid_key else tuple(row.cell_vals[col.id] for col in self.pk_columns)
                if key_vals not in self.updates:
                    continue
                self.matched_key_vals.add(key_vals)
                col_vals = self.updates[key_vals]
                for col, val in col_vals.items():
                    slot_idx = self.col_slot_idxs[col]
                    row[slot_idx] = val
            yield batch

    def unmatched_rows(self) -> list[dict[str, Any]]:
        """Return rows that didn't get used in the updates as a list of dicts compatible with TableVersion.insert()."""
        result: list[dict[str, Any]] = []
        for key_vals, col_vals in self.updates.items():
            if key_vals in self.matched_key_vals:
                continue
            row = {col.name: val for col, val in zip(self.pk_columns, key_vals)}
            row.update({col.name: val for col, val in col_vals.items()})
            result.append(row)
        return result

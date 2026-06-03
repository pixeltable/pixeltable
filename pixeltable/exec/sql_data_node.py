from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator, ClassVar

import sqlalchemy as sql

from pixeltable import catalog, exprs
from pixeltable.utils.local_store import TempStore

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

if TYPE_CHECKING:
    from pixeltable.io.data_sources import SqlDataSource

_logger = logging.getLogger('pixeltable')


class SqlDataNode(ExecNode):
    """
    Streams a SqlDataSource (a normalized SELECT executed against an Engine or Connection) into DataRowBatches.

    Populates the destination table's column slots by matching each SELECT output column to a column of the
    same name; any slot not populated from the source is set to None. The SQL source's columns are assumed to
    have already been validated against the destination schema.
    """

    BATCH_SIZE: ClassVar[int] = 1024

    tbl: catalog.TableVersionHandle
    sql_source: SqlDataSource
    # output_exprs is declared in the superclass, but we redeclare it here with a more specific type
    output_exprs: list[exprs.ColumnRef]
    _owns_conn: bool
    _conn: sql.Connection | None
    _result: sql.CursorResult | None
    _mapped_cols: list[exprs.ColumnSlotIdx]
    _unmapped_slot_idxs: list[int]

    def __init__(
        self, tbl: catalog.TableVersionHandle, sql_source: SqlDataSource, row_builder: exprs.RowBuilder
    ) -> None:
        output_exprs = list(row_builder.input_exprs)
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.get().is_insertable
        self.tbl = tbl
        self.sql_source = sql_source
        self._owns_conn = False
        self._conn = None
        self._result = None
        self._mapped_cols = []
        self._unmapped_slot_idxs = []

    def _open(self) -> None:
        # output_exprs are ColumnRefs to the destination table's insertable columns; index them by name so each
        # SELECT output column can be matched to its destination slot.
        dest_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx) for col_ref in self.output_exprs
        }
        all_output_slot_idxs = {e.slot_idx for e in self.output_exprs}

        # source_names are the SELECT's output column names, positionally aligned with each result row; each is
        # assumed to resolve to a destination column of the same name.
        source_names = [
            getattr(c, 'name', None) or getattr(c, 'key', None) for c in self.sql_source.select_stmt.selected_columns
        ]
        self._mapped_cols = [dest_cols_by_name[n] for n in source_names]
        mapped_slot_idxs = {c.slot_idx for c in self._mapped_cols}
        self._unmapped_slot_idxs = list(all_output_slot_idxs - mapped_slot_idxs)

        if isinstance(self.sql_source.conn, sql.Engine):
            self._conn = self.sql_source.conn.connect()
            self._owns_conn = True
        else:
            assert isinstance(self.sql_source.conn, sql.Connection)
            self._conn = self.sql_source.conn
            self._owns_conn = False

        try:
            self._result = (
                self._conn.execution_options(stream_results=True)
                .execute(  # type: ignore[call-overload]
                    self.sql_source.select_stmt
                )
                .yield_per(self.BATCH_SIZE)
            )
        except BaseException:
            if self._owns_conn:
                self._conn.close()
                self._conn = None
            raise

    async def __aiter__(self) -> AsyncIterator[DataRowBatch]:
        assert self._result is not None
        output_batch = DataRowBatch(self.row_builder)
        for sa_row in self._result:
            output_row = self.row_builder.make_row()
            for col_info, val in zip(self._mapped_cols, sa_row, strict=True):
                col = col_info.col
                if col.col_type.is_image_type() and isinstance(val, bytes):
                    # Mirror InMemoryDataNode: spill image bytes to TempStore and assign the resulting path.
                    # DataRow.__setitem__ only runs on-write media validation when given a path/URL, not bytes.
                    filepath, _ = TempStore.save_media_object(
                        val, col.tbl_handle.id, col.id, col.get_tbl().version, format=None
                    )
                    output_row[col_info.slot_idx] = str(filepath)
                else:
                    output_row[col_info.slot_idx] = val
            for slot_idx in self._unmapped_slot_idxs:
                output_row[slot_idx] = None
            output_batch.add_row(output_row)
            if len(output_batch) >= self.BATCH_SIZE:
                _logger.debug(f'SqlDataNode: yielding batch of {len(output_batch)} rows')
                yield output_batch
                output_batch = DataRowBatch(self.row_builder)

        if len(output_batch) > 0:
            _logger.debug(f'SqlDataNode: yielding final batch of {len(output_batch)} rows')
            yield output_batch

    def _close(self) -> None:
        if self._result is not None:
            self._result.close()
            self._result = None
        if self._conn is not None and self._owns_conn:
            self._conn.close()
        self._conn = None

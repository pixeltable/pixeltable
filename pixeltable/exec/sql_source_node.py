from __future__ import annotations

import logging
from typing import TYPE_CHECKING, AsyncIterator, ClassVar

import sqlalchemy as sql

from pixeltable import catalog, exprs
from pixeltable.utils.sql import selectable_columns

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

if TYPE_CHECKING:
    from pixeltable.io.data_sources import SqlDataSource

_logger = logging.getLogger('pixeltable')


class SqlSourceNode(ExecNode):
    """
    Streams a SqlDataSource (a SQLAlchemy Selectable executed against an Engine or Connection) into DataRowBatches.

    Same output contract as InMemoryDataNode:
      - output_exprs = row_builder.input_exprs
      - populates user-column slots by name; sets unmapped slots to None.

    Source/destination column compatibility (unnamed columns, duplicates, computed-column collisions,
    missing required columns, unknown destination columns) is validated by the caller (eg, `import_sql`)
    before the plan is constructed; this node only builds the runtime slot mapping.
    """

    BATCH_SIZE: ClassVar[int] = 1024

    tbl: catalog.TableVersionHandle
    sql_source: SqlDataSource
    output_exprs: list[exprs.ColumnRef]

    def __init__(
        self, tbl: catalog.TableVersionHandle, sql_source: SqlDataSource, row_builder: exprs.RowBuilder
    ) -> None:
        output_exprs = list(row_builder.input_exprs)
        super().__init__(row_builder, output_exprs, [], None)
        assert tbl.get().is_insertable
        self.tbl = tbl
        self.sql_source = sql_source
        self._owns_conn = False
        self._conn: sql.Connection | None = None
        self._result: sql.CursorResult | None = None
        self._mapped_slot_idxs: list[int] = []
        self._unmapped_slot_idxs: list[int] = []

    def _open(self) -> None:
        user_cols_by_name = {
            col_ref.col.name: col_ref.slot_idx for col_ref in self.output_exprs if col_ref.col.name is not None
        }
        all_output_slot_idxs = {e.slot_idx for e in self.output_exprs}

        sa_cols = selectable_columns(self.sql_source.selectable)
        source_names = [getattr(c, 'name', None) or getattr(c, 'key', None) for c in sa_cols]
        # Caller (eg, import_sql) is responsible for validating that all source names are non-None and resolve
        # to a destination column.
        self._mapped_slot_idxs = [user_cols_by_name[n] for n in source_names]
        self._unmapped_slot_idxs = list(all_output_slot_idxs - set(self._mapped_slot_idxs))

        if isinstance(self.sql_source.conn, sql.Engine):
            self._conn = self.sql_source.conn.connect()
            self._owns_conn = True
        else:
            self._conn = self.sql_source.conn
            self._owns_conn = False

        # Tables, subqueries, and aliases are not directly executable; wrap them in `select(...)`.
        src = self.sql_source.selectable
        stmt: sql.Executable = src if isinstance(src, sql.Executable) else sql.select(src)  # type: ignore[call-overload]
        try:
            self._result = self._conn.execute(stmt)
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
            for slot_idx, val in zip(self._mapped_slot_idxs, sa_row, strict=True):
                output_row[slot_idx] = val
            for slot_idx in self._unmapped_slot_idxs:
                output_row[slot_idx] = None
            output_batch.add_row(output_row)
            if len(output_batch) == self.BATCH_SIZE:
                _logger.debug(f'SqlSourceNode: yielding batch of {len(output_batch)} rows')
                yield output_batch
                output_batch = DataRowBatch(self.row_builder)

        if len(output_batch) > 0:
            _logger.debug(f'SqlSourceNode: yielding final batch of {len(output_batch)} rows')
            yield output_batch

    def _close(self) -> None:
        if self._result is not None:
            self._result.close()
            self._result = None
        if self._conn is not None and self._owns_conn:
            self._conn.close()
        self._conn = None

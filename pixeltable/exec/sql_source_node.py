from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import AsyncIterator, ClassVar

import sqlalchemy as sql

from pixeltable import catalog, exceptions as excs, exprs

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode

_logger = logging.getLogger('pixeltable')


@dataclass
class SqlDataSource:
    """A user-supplied SQL source: a SQLAlchemy `Selectable` and an `Engine` or `Connection` to run it against."""

    selectable: sql.Selectable
    conn: sql.Engine | sql.Connection


def _selectable_columns(selectable: sql.Selectable) -> list[sql.ColumnElement]:
    """Return the output columns of a Selectable in their SELECT-clause order."""
    if hasattr(selectable, 'selected_columns'):
        # Select / TextualSelect / CompoundSelect
        return list(selectable.selected_columns)
    # Table / Subquery / Alias
    return list(selectable.columns)  # type: ignore[attr-defined]


class SqlSourceNode(ExecNode):
    """
    Streams a SqlDataSource (a SQLAlchemy Selectable executed against an Engine or Connection) into DataRowBatches.

    Same output contract as InMemoryDataNode:
      - output_exprs = row_builder.input_exprs
      - populates user-column slots by name; sets unmapped slots to None.
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
        self._all_output_slot_idxs: set[int] = set()
        self._mapped_slot_idxs: list[int] = []

    def _open(self) -> None:
        # Collect destination-side metadata
        tbl_version = self.tbl.get()
        all_cols_by_name = tbl_version.cols_by_name
        computed_col_names = {name for name, col in all_cols_by_name.items() if col.is_computed}
        required_col_names = {name for name, col in all_cols_by_name.items() if col.is_required_for_insert}

        user_cols_by_name = {
            col_ref.col.name: exprs.ColumnSlotIdx(col_ref.col, col_ref.slot_idx)
            for col_ref in self.output_exprs
            if col_ref.col.name is not None
        }
        self._all_output_slot_idxs = {e.slot_idx for e in self.output_exprs}

        tbl_name = tbl_version.name
        sa_cols = _selectable_columns(self.sql_source.selectable)
        source_names: list[str] = []
        for i, c in enumerate(sa_cols):
            name = getattr(c, 'name', None) or getattr(c, 'key', None)
            if name is None:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'SQL source has an unnamed output column at position {i} (when inserting into table '
                    f"{tbl_name!r}); alias it via `expr.label('name')` so it can be matched to a Pixeltable column.",
                )
            source_names.append(name)

        seen: set[str] = set()
        for name in source_names:
            if name in seen:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_SCHEMA,
                    f'SQL source has duplicate output column {name!r} (when inserting into table {tbl_name!r}); '
                    f'output column names must be unique.',
                )
            seen.add(name)
            if name in computed_col_names:
                raise excs.RequestError(
                    excs.ErrorCode.UNSUPPORTED_OPERATION,
                    f'SQL source column {name!r} maps to computed column {name!r} in destination table '
                    f'{tbl_name!r}; computed columns are populated automatically and cannot receive values.',
                )
            if name not in all_cols_by_name:
                raise excs.NotFoundError(
                    excs.ErrorCode.COLUMN_NOT_FOUND,
                    f'SQL source column {name!r} does not match any column in destination table {tbl_name!r}.',
                )

        missing_cols = required_col_names - seen
        if len(missing_cols) > 0:
            raise excs.RequestError(
                excs.ErrorCode.MISSING_REQUIRED,
                f'Destination table {tbl_name!r} has required column(s) ({", ".join(sorted(missing_cols))}) '
                f'that are not provided by the SQL source.',
            )

        self._mapped_slot_idxs = [user_cols_by_name[n].slot_idx for n in source_names]

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
        unmapped_slot_idxs = list(self._all_output_slot_idxs - set(self._mapped_slot_idxs))

        output_batch = DataRowBatch(self.row_builder)
        for sa_row in self._result:
            output_row = self.row_builder.make_row()
            for slot_idx, val in zip(self._mapped_slot_idxs, sa_row):
                output_row[slot_idx] = val
            for slot_idx in unmapped_slot_idxs:
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

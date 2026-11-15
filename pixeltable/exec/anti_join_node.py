from __future__ import annotations

import contextlib
from typing import Any, AsyncGenerator

import sqlalchemy as sql

from pixeltable import catalog, exprs
from pixeltable.index.btree import BtreeIndex
from pixeltable.runtime import get_runtime

from .data_row_batch import DataRowBatch
from .exec_node import ExecNode
from .sql_node import SqlNode

# bind parameters per key lookup statement; Postgres allows at most 65535, and the from clause binds some as well
_KEY_LOOKUP_BIND_PARAMS = 10_000


class AntiJoinNode(ExecNode):
    """
    Anti-joins the input rows with tbl on the primary key: drops each row whose key is already in tbl or in an earlier
    input row, and counts the dropped rows in ExecContext.num_skipped_rows.
    """

    tbl: catalog.TableVersionHandle
    pk_cols: list[catalog.Column]
    pk_slot_idxs: list[int]
    # per key column: True if the unique index stores a prefix of the string, as for a data-versioned table
    pk_col_is_prefix: list[bool]
    keys_per_lookup: int
    seen_keys: set[tuple[Any, ...]]

    def __init__(self, tbl: catalog.TableVersionHandle, row_builder: exprs.RowBuilder, input: ExecNode):
        super().__init__(row_builder, input.output_exprs, input.output_exprs, input)
        self.tbl = tbl
        self.pk_cols = tbl.get().primary_key_columns()
        assert len(self.pk_cols) > 0
        self.pk_slot_idxs = [row_builder.table_columns[col] for col in self.pk_cols]
        is_data_versioned = tbl.get().is_data_versioned
        self.pk_col_is_prefix = [is_data_versioned and col.col_type.is_string_type() for col in self.pk_cols]
        # a lookup binds one parameter per key column per key
        self.keys_per_lookup = _KEY_LOOKUP_BIND_PARAMS // len(self.pk_cols)
        self.seen_keys = set()

    async def __aiter__(self) -> AsyncGenerator[DataRowBatch, None]:
        # input rows and their keys, for the next lookup
        pending: list[tuple[tuple[Any, ...] | None, exprs.DataRow]] = []
        num_pending_keys = 0
        async with contextlib.aclosing(aiter(self.input)) as input_iter:
            async for batch in input_iter:
                for row in batch:
                    key = self._key(row)
                    pending.append((key, row))
                    if key is None:
                        continue
                    num_pending_keys += 1
                    if num_pending_keys == self.keys_per_lookup:
                        output_batch = self._new_rows(pending)
                        pending, num_pending_keys = [], 0
                        if len(output_batch) > 0:
                            yield output_batch
        if len(pending) > 0:
            output_batch = self._new_rows(pending)
            if len(output_batch) > 0:
                yield output_batch

    def _key(self, row: exprs.DataRow) -> tuple[Any, ...] | None:
        """Return the primary key of row, or None if a key column has an error or is null."""
        if any(row.has_exc(slot_idx) or row[slot_idx] is None for slot_idx in self.pk_slot_idxs):
            return None
        return tuple(
            row[slot_idx][: BtreeIndex.MAX_STRING_LEN] if is_prefix else row[slot_idx]
            for slot_idx, is_prefix in zip(self.pk_slot_idxs, self.pk_col_is_prefix)
        )

    def _new_rows(self, rows: list[tuple[tuple[Any, ...] | None, exprs.DataRow]]) -> DataRowBatch:
        """Return the rows whose key is neither in tbl nor in an earlier row, and the rows without a key."""
        assert self.ctx is not None
        stored_keys = self._stored_keys([key for key, _ in rows if key is not None])
        output_batch = DataRowBatch(self.row_builder)
        for key, row in rows:
            if key is not None:
                if key in stored_keys or key in self.seen_keys:
                    self.ctx.num_skipped_rows += 1
                    continue
                self.seen_keys.add(key)
            output_batch.add_row(row)
        return output_batch

    def _stored_keys(self, keys: list[tuple[Any, ...]]) -> set[tuple[Any, ...]]:
        """Return the keys that are in tbl, with a single lookup."""
        assert len(keys) <= self.keys_per_lookup
        if len(keys) == 0:
            return set()
        sa_key_cols = [
            sql.func.left(col.sa_col, BtreeIndex.MAX_STRING_LEN) if is_prefix else col.sa_col
            for col, is_prefix in zip(self.pk_cols, self.pk_col_is_prefix)
        ]
        stmt = sql.select(*sa_key_cols).where(sql.tuple_(*sa_key_cols).in_(keys))
        # restricts the lookup to the rows visible at the current version
        stmt = SqlNode.create_from_clause(self.tbl.get().path, stmt)
        return {tuple(row) for row in get_runtime().conn.execute(stmt)}

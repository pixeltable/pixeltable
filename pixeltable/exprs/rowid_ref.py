from __future__ import annotations

import logging
from typing import Any, Optional, cast
from uuid import UUID

import sqlalchemy as sql

from pixeltable import catalog, type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

_logger = logging.getLogger('pixeltable')


class RowidRef(Expr):
    """A reference to a part of a table rowid

    This is used internally to support grouping by a base table and for references to the 'pos' column.
    When a RowidRef is part of a computed column in a view, the view's TableVersion isn't available when
    _from_dict()/init() is called, which is why this class effectively has two separate paths for construction
    (with and without a TableVersion).
    """

    tbl: Optional[catalog.TableVersionHandle]
    normalized_base: Optional[catalog.TableVersionHandle]
    tbl_id: UUID
    normalized_base_id: UUID
    rowid_component_idx: int

    def __init__(
        self,
        tbl: Optional[catalog.TableVersionHandle],
        idx: int,
        tbl_id: Optional[UUID] = None,
        normalized_base_id: Optional[UUID] = None,
    ):
        super().__init__(ts.IntType(nullable=False))
        self.tbl = tbl
        if tbl is not None:
            # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
            # (which has the same values as all its descendent views)
            normalized_base = tbl
            # don't try to reference tbl.store_tbl here
            while normalized_base.get().base is not None and normalized_base.get().base.get().num_rowid_columns() > idx:
                normalized_base = normalized_base.get().base
            self.normalized_base = normalized_base
        else:
            self.normalized_base = None

        # if we're initialized by _from_dict(), we only have the ids, not the TableVersion itself
        self.tbl_id = tbl.id if tbl is not None else tbl_id
        self.normalized_base_id = self.normalized_base.id if self.normalized_base is not None else normalized_base_id
        self.rowid_component_idx = idx
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: RowidRef) -> bool:
        return (
            self.normalized_base_id == other.normalized_base_id
            and self.rowid_component_idx == other.rowid_component_idx
        )

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [
            *super()._id_attrs(),
            ('normalized_base_id', self.normalized_base_id),
            ('idx', self.rowid_component_idx),
        ]

    def __repr__(self) -> str:
        # check if this is the pos column of a component view
        from pixeltable import store

        tbl = self.tbl.get() if self.tbl is not None else catalog.Catalog.get().get_tbl_version(self.tbl_id, None)
        if (
            tbl.is_component_view
            and self.rowid_component_idx == cast(store.StoreComponentView, tbl.store_tbl).pos_col_idx
        ):
            return catalog.globals._POS_COLUMN_NAME
        return ''

    def set_tbl(self, tbl: catalog.TableVersionPath) -> None:
        """Change the table that is being referenced.
        This can be necessary during query planning, because at that stage we try to minimize the total number of
        tables that are referenced/need to be joined.
        We can only change to a view of the original table (which shares the base's rowid columns).
        """
        if self.tbl_id == tbl.tbl_version.id:
            return
        base_ids = [tbl_version.id for tbl_version in tbl.get_tbl_versions()]
        assert self.tbl_id in base_ids  # our current TableVersion is a base of the new TableVersion
        self.tbl = tbl.tbl_version
        self.tbl_id = self.tbl.id

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        tbl = self.tbl.get() if self.tbl is not None else catalog.Catalog.get().get_tbl_version(self.tbl_id, None)
        assert tbl.is_validated
        rowid_cols = tbl.store_tbl.rowid_columns()
        assert self.rowid_component_idx <= len(rowid_cols), (
            f'{self.rowid_component_idx} not consistent with {rowid_cols}'
        )
        return rowid_cols[self.rowid_component_idx]

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row.pk[self.rowid_component_idx]

    def _as_dict(self) -> dict:
        return {
            'tbl_id': str(self.tbl_id),
            'normalized_base_id': str(self.normalized_base_id),
            'idx': self.rowid_component_idx,
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> RowidRef:
        tbl_id, normalized_base_id, idx = UUID(d['tbl_id']), UUID(d['normalized_base_id']), d['idx']
        return cls(tbl=None, idx=idx, tbl_id=tbl_id, normalized_base_id=normalized_base_id)

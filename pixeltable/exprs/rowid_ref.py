from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
from uuid import UUID

import sqlalchemy as sql

from .expr import Expr
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.type_system as ts
import pixeltable.catalog as catalog


class RowidRef(Expr):
    """A reference to a part of a table rowid

    This is used internally to support grouping by a base table and for references to the 'pos' column.
    When a RowidRef is part of a computed column in a view, the view's TableVersion isn't available when
    _from_dict()/init() is called, which is why this class effectively has two separate paths for construction
    (with and without a TableVersion).
    """
    def __init__(
            self, tbl: catalog.TableVersion, idx: int,
            tbl_id: Optional[UUID] = None, normalized_base_id: Optional[UUID] = None):
        super().__init__(ts.IntType(nullable=False))
        self.tbl = tbl
        if tbl is not None:
            # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
            # (which has the same values as all its descendent views)
            normalized_base = tbl
            # don't try to reference tbl.store_tbl here
            while normalized_base.base is not None and normalized_base.base.num_rowid_columns() > idx:
                normalized_base = normalized_base.base
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
        return self.normalized_base_id == other.normalized_base_id \
            and self.rowid_component_idx == other.rowid_component_idx

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() +\
            [('normalized_base_id', self.normalized_base_id), ('idx', self.rowid_component_idx)]

    def __str__(self) -> str:
        # check if this is the pos column of a component view
        tbl = self.tbl if self.tbl is not None else catalog.Catalog.get().tbl_versions[(self.tbl_id, None)]
        if tbl.is_component_view() and self.rowid_component_idx == tbl.store_tbl.pos_col_idx:
            return catalog.globals.POS_COLUMN_NAME
        return ''

    def set_tbl(self, tbl: catalog.TableVersionPath) -> None:
        """Change the table that is being referenced.
        This can be necessary during query planning, because at that stage we try to minimize the total number of
        tables that are referenced/need to be joined.
        We can only change to a view of the original table (which shares the base's rowid columns).
        """
        if self.tbl_id == tbl.tbl_version.id:
            return
        tbl_version_ids = [tbl_version.id for tbl_version in tbl.get_tbl_versions()]
        assert self.tbl_id in tbl_version_ids
        self.tbl = tbl.tbl_version
        self.tbl_id = self.tbl.id

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        tbl = self.tbl if self.tbl is not None else catalog.Catalog.get().tbl_versions[(self.tbl_id, None)]
        rowid_cols = tbl.store_tbl.rowid_columns()
        return rowid_cols[self.rowid_component_idx]

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row.pk[self.rowid_component_idx]

    def _as_dict(self) -> Dict:
        return {
            'tbl_id': str(self.tbl_id),
            'normalized_base_id': str(self.normalized_base_id),
            'idx': self.rowid_component_idx,
        }

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        tbl_id, normalized_base_id, idx = UUID(d['tbl_id']), UUID(d['normalized_base_id']), d['idx']
        return cls(tbl=None, idx=idx, tbl_id=tbl_id, normalized_base_id=normalized_base_id)


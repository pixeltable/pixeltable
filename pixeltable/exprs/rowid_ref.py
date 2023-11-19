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
    """A reference to (a part of) a table rowid column

    This is used internally to support grouping by a base table and for references to the 'pos' column.
    """
    def __init__(self, tbl: catalog.TableVersion, idx: int):
        super().__init__(ts.IntType(nullable=False))
        # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
        # (which has the same values as all its descendent views)
        while tbl.base is not None and len(tbl.base.store_tbl.rowid_columns()) > idx:
            tbl = tbl.base
        self.tbl = tbl
        self.rowid_component_idx = idx
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: RowidRef) -> bool:
        return self.tbl is other.tbl and self.rowid_component_idx == other.rowid_component_idx

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('tbl_id', self.tbl.id), ('idx', self.rowid_component_idx)]

    def __str__(self) -> str:
        # check if this is the pos column of a component view
        if self.tbl.is_component_view() and self.rowid_component_idx == self.tbl.store_tbl.pos_col_idx:
            return catalog.globals.POS_COLUMN_NAME
        return None

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        rowid_cols = self.tbl.store_tbl.rowid_columns()
        return rowid_cols[self.rowid_component_idx]

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row.pk[self.rowid_component_idx]

    def _as_dict(self) -> Dict:
        return {'tbl_id': str(self.tbl.id), 'idx': self.rowid_component_idx}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr], t: catalog.TableVersion) -> Expr:
        tbl = t.find_tbl(UUID(d['tbl_id']))
        assert tbl is not None
        return cls(tbl, d['idx'])


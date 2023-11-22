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
    """
    def __init__(self, tbl: catalog.TableVersion, idx: int):
        super().__init__(ts.IntType(nullable=False))
        self.tbl = tbl
        # normalize to simplify comparisons: we refer to the lowest base table that has the requested rowid idx
        # (which has the same values as all its descendent views)
        normalized_base = tbl
        while normalized_base.base is not None and len(normalized_base.base.store_tbl.rowid_columns()) > idx:
            normalized_base = normalized_base.base
        self.normalized_base = normalized_base
        self.rowid_component_idx = idx
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self)

    def _equals(self, other: RowidRef) -> bool:
        return self.normalized_base is other.normalized_base and self.rowid_component_idx == other.rowid_component_idx

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() +\
            [('normalized_base_id', self.normalized_base.id), ('idx', self.rowid_component_idx)]

    def __str__(self) -> str:
        # check if this is the pos column of a component view
        if self.tbl.is_component_view() and self.rowid_component_idx == self.tbl.store_tbl.pos_col_idx:
            return catalog.globals.POS_COLUMN_NAME
        return None

    def set_tbl(self, tbl: catalog.TableVersion) -> None:
        """Change the table that is being referenced.
        This can be necessary during query planning, because at that stage we try to minimize the total number of
        tables that are referenced/need to be joined.
        We can only change to a view of the original table (which shares the base's rowid columns).
        """
        if tbl is self.tbl:
            return
        bases = tbl.get_bases()
        assert self.tbl in bases
        self.tbl = tbl

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


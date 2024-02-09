from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
import enum

import sqlalchemy as sql

from .expr import Expr
from .column_ref import ColumnRef
from .row_builder import RowBuilder
from .data_row import DataRow
import pixeltable.catalog as catalog
import pixeltable.type_system as ts


class ColumnPropertyRef(Expr):
    """A reference to a property of a table column

    The properties themselves are type-specific and may or may not need to reference the underlying column data.
    """
    class Property(enum.Enum):
        ERRORTYPE = 0
        ERRORMSG = 1
        FILEURL = 2
        LOCALPATH = 3

    def __init__(self, col_ref: ColumnRef, prop: Property):
        super().__init__(ts.StringType(nullable=True))
        self.components = [col_ref]
        self.prop = prop
        self.id = self._create_id()

    def default_column_name(self) -> Optional[str]:
        return str(self).replace('.', '_')

    def _equals(self, other: ColumnRef) -> bool:
        return self.prop == other.prop

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('prop', self.prop.value)]

    @property
    def _col_ref(self) -> ColumnRef:
        return self.components[0]

    def __str__(self) -> str:
        return f'{self._col_ref}.{self.prop.name.lower()}'

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        if not self._col_ref.col.is_stored:
            return None
        if self.prop == self.Property.ERRORTYPE:
            assert self._col_ref.col.sa_errortype_col is not None
            return self._col_ref.col.sa_errortype_col
        if self.prop == self.Property.ERRORMSG:
            assert self._col_ref.col.sa_errormsg_col is not None
            return self._col_ref.col.sa_errormsg_col
        if self.prop == self.Property.FILEURL:
            # the file url is stored as the column value
            return self._col_ref.sql_expr()
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        assert self.prop == self.Property.FILEURL or self.prop == self.Property.LOCALPATH
        assert data_row.has_val[self._col_ref.slot_idx]
        if self.prop == self.Property.FILEURL:
            data_row[self.slot_idx] = data_row.file_urls[self._col_ref.slot_idx]
        if self.prop == self.Property.LOCALPATH:
            data_row[self.slot_idx] = data_row.file_paths[self._col_ref.slot_idx]

    def _as_dict(self) -> Dict:
        return {'prop': self.prop.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'prop' in d
        return cls(components[0], cls.Property(d['prop']))


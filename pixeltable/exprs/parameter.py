from __future__ import annotations
from typing import List, Tuple, Any, Optional, Dict

import sqlalchemy as sql

from .expr import Expr
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.type_system as ts


class Parameter(Expr):
    """An expr parameter, needed for ExprTemplates

    Parameters have a name and type and need to have been replaced by actual expressions before evaluation.
    """

    def __init__(self, name: str, col_type: ts.ColumnType):
        super().__init__(col_type)
        self.name = name

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('name', self.name)]

    def default_column_name(self) -> Optional[str]:
        assert False
        return None

    def _equals(self, other: Parameter) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        assert False
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        assert False

    def _as_dict(self) -> Dict:
        return {'name': self.name, 'type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, _: List[Expr]) -> Expr:
        return cls(d['name'], ts.ColumnType.from_dict(d['type']))

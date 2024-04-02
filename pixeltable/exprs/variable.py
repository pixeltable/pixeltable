from __future__ import annotations

from typing import List, Tuple, Any, Dict, NoReturn

import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder


class Variable(Expr):
    """An expr parameter, needed for ExprTemplateFunctions

    A Variable has a name and type and needs to have been replaced by an actual expression before evaluation.
    """

    def __init__(self, name: str, col_type: ts.ColumnType):
        super().__init__(col_type)
        self.name = name
        self.id = self._create_id()

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('name', self.name)]

    def default_column_name(self) -> NoReturn:
        raise NotImplementedError()

    def _equals(self, other: Variable) -> bool:
        return self.name == other.name

    def __str__(self) -> str:
        return self.name

    def sql_expr(self) -> NoReturn:
        raise NotImplementedError()

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> NoReturn:
        raise NotImplementedError()

    def _as_dict(self) -> Dict:
        return {'name': self.name, 'type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, _: List[Expr]) -> Expr:
        return cls(d['name'], ts.ColumnType.from_dict(d['type']))

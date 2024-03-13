import json
from typing import Optional, Dict, List, Tuple, Any

import sqlalchemy as sql

import pixeltable.type_system as ts
from .expr import DataRow, Expr
from .row_builder import RowBuilder


class TypeCast(Expr):
    """
    An `Expr` that represents a type conversion from an underlying `Expr` to
    a specified `ColumnType`.
    """
    def __init__(self, underlying: Expr, new_type: ts.ColumnType):
        super().__init__(new_type)
        self.components: List[Expr] = [underlying]
        self.id: Optional[int] = self._create_id()

    @property
    def _underlying(self):
        return self.components[0]

    def _equals(self, other: 'TypeCast') -> bool:
        # `TypeCast` has no properties beyond those captured by `Expr`.
        return True

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('new_type', self.col_type)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        """
        `sql_expr` is unimplemented for now, in order to sidestep potentially thorny
        questions about consistency of doing type conversions in both Python and Postgres.
        """
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        original_val = data_row[self._underlying.slot_idx]
        data_row[self.slot_idx] = self.col_type.create_literal(original_val)

    def _as_dict(self) -> Dict:
        return {'new_type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'new_type' in d
        assert len(components) == 1
        return cls(components[0], ts.ColumnType.from_dict(d['new_type']))

    def __str__(self) -> str:
        return f'{self._underlying}.astype({self.col_type})'

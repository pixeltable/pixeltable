from typing import Optional

import sqlalchemy as sql

from pixeltable import type_system as ts

from .expr import DataRow, Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class TypeCast(Expr):
    """
    An `Expr` that represents a type conversion from an underlying `Expr` to
    a specified `ColumnType`.
    """

    def __init__(self, underlying: Expr, new_type: ts.ColumnType):
        super().__init__(new_type)
        self.components: list[Expr] = [underlying]
        self.id: Optional[int] = self._create_id()

    def _equals(self, other: 'TypeCast') -> bool:
        # `TypeCast` has no properties beyond those captured by `Expr`.
        return True

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        """
        sql_expr() is unimplemented for now, in order to sidestep potentially thorny
        questions about consistency of doing type conversions in both Python and Postgres.
        """
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        original_val = data_row[self._op1.slot_idx]
        data_row[self.slot_idx] = self.col_type.create_literal(original_val)

    def as_literal(self) -> Optional[Literal]:
        op1_lit = self._op1.as_literal()
        if op1_lit is None:
            return None
        if not (
            self.col_type.is_numeric_type() and (op1_lit.col_type.is_numeric_type() or op1_lit.col_type.is_bool_type())
        ):
            return None

        op1_val = op1_lit.val
        if self.col_type.is_int_type():
            return Literal(int(op1_val), self.col_type)
        elif self.col_type.is_float_type():
            return Literal(float(op1_val), self.col_type)
        return None

    def _as_dict(self) -> dict:
        return {'new_type': self.col_type.as_dict(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> 'TypeCast':
        assert 'new_type' in d
        assert len(components) == 1
        return cls(components[0], ts.ColumnType.from_dict(d['new_type']))

    def __repr__(self) -> str:
        return f'{self._op1}.astype({self.col_type._to_str(as_schema=True)})'

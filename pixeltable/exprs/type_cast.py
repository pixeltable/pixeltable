from typing import Optional

import sqlalchemy as sql

from pixeltable.exprs import DataRow, Expr, RowBuilder
from pixeltable.type_system import ColumnType


class TypeCast(Expr):

    def __init__(self, underlying: Expr, new_type: ColumnType):
        super().__init__(new_type)
        self._underlying = underlying

    def _equals(self, other: Expr) -> bool:
        if isinstance(other, TypeCast):
            return self._underlying == other._underlying and self.col_type == other.col_type
        else:
            return False

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        underlying_sql_expr = self._underlying.sql_expr()
        if underlying_sql_expr is None:
            return None
        else:
            return underlying_sql_expr.cast(self.col_type.to_sa_type)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        original_val = data_row[self._underlying.slot_idx]
        if self.col_type.is_string_type():
            cast_val = str(original_val)
        elif self.col_type.is_json_type() and self._underlying.col_type.is_string_type():
            pass
        else:
            raise RuntimeError(f'Expression of type `{self._underlying.col_type}` cannot be cast to `{self.col_type}`')
        data_row[self.slot_idx] = cast_val

    def __str__(self) -> str:
        return f'{self._underlying}.astype({self.col_type})'

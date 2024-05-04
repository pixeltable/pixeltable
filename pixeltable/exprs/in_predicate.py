from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple, Callable
import operator

import sqlalchemy as sql

from .expr import Expr
from .globals import LogicalOperator
from .predicate import Predicate
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.exceptions as excs


class InPredicate(Predicate):
    def __init__(self, lhs: Expr, value_list: list[Any]):
        if not lhs.col_type.is_scalar_type():
            raise excs.Error(f'isin(): only supported for scalar types, not {lhs.col_type}')
        super().__init__()
        self.components = [lhs]
        for val in value_list:
            try:
                lhs.col_type.validate_literal(val)
            except TypeError:
                raise excs.Error(
                    f'isin(): list item {val!r} is not compatible with the type of {lhs}, which is {lhs.col_type}')
        self.value_list = value_list

        self.id = self._create_id()

    def __str__(self) -> str:
        return f'{self.components[0]}.isin({self.value_list})'

    def _equals(self, other: InPredicate) -> bool:
        return self.value_list == other.value_list

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('value_list', self.value_list)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        lhs_sql_exprs = self.components[0].sql_expr()
        if lhs_sql_exprs is None:
            return None
        return lhs_sql_exprs.in_(self.value_list)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        lhs_val = data_row[self.components[0].slot_idx]
        data_row[self.slot_idx] = lhs_val in self.value_list

    def _as_dict(self) -> Dict:
        return {'value_list': self.value_list, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'value_list' in d
        return cls(components[0], d['value_list'])


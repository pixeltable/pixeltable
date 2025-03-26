from __future__ import annotations

from typing import Optional

import sqlalchemy as sql

import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class IsNull(Expr):
    def __init__(self, e: Expr):
        super().__init__(ts.BoolType())
        self.components = [e]
        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'{self.components[0]} == None'

    def _equals(self, other: IsNull) -> bool:
        return True

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        e = sql_elements.get(self.components[0])
        if e is None:
            return None
        return e == None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = data_row[self.components[0].slot_idx] is None

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> IsNull:
        assert len(components) == 1
        return cls(components[0])

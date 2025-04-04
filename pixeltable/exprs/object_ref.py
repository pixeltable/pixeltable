from __future__ import annotations

from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr, ExprScope
from .json_mapper import JsonMapperDispatch
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class ObjectRef(Expr):
    """
    Reference to an intermediate result, such as the "scope variable" produced by a JsonMapper.
    The object is generated/materialized elsewhere and establishes a new scope.
    """

    def __init__(self, scope: ExprScope, owner: JsonMapperDispatch):
        # TODO: do we need an Unknown type after all?
        super().__init__(ts.JsonType())  # JsonType: this could be anything
        self._scope = scope
        self.owner = owner
        self.id = self._create_id()

    def _id_attrs(self) -> list[tuple[str, Any]]:
        # We have no components, so we can't rely on the default behavior here (otherwise, all ObjectRef
        # instances will be conflated into a single slot).
        return [('addr', id(self))]

    def substitute(self, subs: dict[Expr, Expr]) -> Expr:
        # Just return self; we need to avoid creating a new id after doing the substitution, because otherwise
        # we'll wind up in a situation where the scope_anchor of the enclosing JsonMapper is different from the
        # nested ObjectRefs inside its target_expr (and therefore occupies a different slot_idx).
        return self

    def scope(self) -> ExprScope:
        return self._scope

    def _equals(self, other: ObjectRef) -> bool:
        return self.id == other.id

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, but the value has already been materialized elsewhere
        pass

    def __repr__(self) -> str:
        return f'ObjectRef({self.owner}, {self.id}, {self.owner.id})'

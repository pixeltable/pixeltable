from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
import copy

import sqlalchemy as sql

from .expr import Expr, ExprScope
from .json_mapper import JsonMapper
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.type_system as ts


class ObjectRef(Expr):
    """
    Reference to an intermediate result, such as the "scope variable" produced by a JsonMapper.
    The object is generated/materialized elsewhere and establishes a new scope.
    """
    def __init__(self, scope: ExprScope, owner: JsonMapper):
        # TODO: do we need an Unknown type after all?
        super().__init__(ts.JsonType())  # JsonType: this could be anything
        self._scope = scope
        self.owner = owner
        self.id = self._create_id()

    def scope(self) -> ExprScope:
        return self._scope

    def __str__(self) -> str:
        assert False

    def _equals(self, other: ObjectRef) -> bool:
        return self.owner is other.owner

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, but the value has already been materialized elsewhere
        pass


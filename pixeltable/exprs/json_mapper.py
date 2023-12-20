from __future__ import annotations
from typing import Optional, List, Dict

import sqlalchemy as sql

from .expr import Expr, ExprScope, _GLOBAL_SCOPE
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.catalog as catalog
import pixeltable.type_system as ts


class JsonMapper(Expr):
    """
    JsonMapper transforms the list output of a JsonPath by applying a target expr to every element of the list.
    The target expr would typically contain relative JsonPaths, which are bound to an ObjectRef, which in turn
    is populated by JsonMapper.eval(). The JsonMapper effectively creates a new scope for its target expr.
    """
    def __init__(self, src_expr: Expr, target_expr: Expr):
        # TODO: type spec should be List[target_expr.col_type]
        super().__init__(ts.JsonType())

        # we're creating a new scope, but we don't know yet whether this is nested within another JsonMapper;
        # this gets resolved in bind_rel_paths(); for now we assume we're in the global scope
        self.target_expr_scope = ExprScope(_GLOBAL_SCOPE)

        from .object_ref import ObjectRef
        scope_anchor = ObjectRef(self.target_expr_scope, self)
        self.components = [src_expr, target_expr, scope_anchor]
        self.parent_mapper: Optional[JsonMapper] = None
        self.target_expr_eval_ctx: Optional[RowBuilder.EvalCtx] = None
        self.id = self._create_id()

    def bind_rel_paths(self, mapper: Optional[JsonMapper]) -> None:
        self._src_expr.bind_rel_paths(mapper)
        self._target_expr.bind_rel_paths(self)
        self.parent_mapper = mapper
        parent_scope = _GLOBAL_SCOPE if mapper is None else mapper.target_expr_scope
        self.target_expr_scope.parent = parent_scope

    def scope(self) -> ExprScope:
        # need to ignore target_expr
        return self._src_expr.scope()

    def dependencies(self) -> List[Expr]:
        result = [self._src_expr]
        result.extend(self._target_dependencies(self._target_expr))
        return result

    def _target_dependencies(self, e: Expr) -> List[Expr]:
        """
        Return all subexprs of e of which the scope isn't contained in target_expr_scope.
        Those need to be evaluated before us.
        """
        expr_scope = e.scope()
        if not expr_scope.is_contained_in(self.target_expr_scope):
            return [e]
        result: List[Expr] = []
        for c in e.components:
            result.extend(self._target_dependencies(c))
        return result

    def equals(self, other: Expr) -> bool:
        """
        We override equals() because we need to avoid comparing our scope anchor.
        """
        if type(self) != type(other):
            return False
        return self._src_expr.equals(other._src_expr) and self._target_expr.equals(other._target_expr)

    def __str__(self) -> str:
        return f'{str(self._src_expr)} >> {str(self._target_expr)}'

    @property
    def _src_expr(self) -> Expr:
        return self.components[0]

    @property
    def _target_expr(self) -> Expr:
        return self.components[1]

    @property
    def scope_anchor(self) -> Expr:
        return self.components[2]

    def _equals(self, other: JsonMapper) -> bool:
        return True

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # this will be called, but the value has already been materialized elsewhere
        src = data_row[self._src_expr.slot_idx]
        if not isinstance(src, list):
            # invalid/non-list src path
            data_row[self.slot_idx] = None
            return

        result = [None] * len(src)
        if self.target_expr_eval_ctx is None:
            self.target_expr_eval_ctx = row_builder.create_eval_ctx([self._target_expr])
        for i, val in enumerate(src):
            data_row[self.scope_anchor.slot_idx] = val
            # stored target_expr
            exc_tb = row_builder.eval(data_row, self.target_expr_eval_ctx)
            assert exc_tb is None
            result[i] = data_row[self._target_expr.slot_idx]
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        """
        We need to avoid serializing component[2], which is an ObjectRef.
        """
        return {'components': [c.as_dict() for c in self.components[0:2]]}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert len(components) == 2
        return cls(components[0], components[1])


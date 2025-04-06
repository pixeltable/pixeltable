from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import sqlalchemy as sql

import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import _GLOBAL_SCOPE, Expr, ExprScope
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from .object_ref import ObjectRef


class JsonMapper(Expr):
    """
    JsonMapper transforms the list output of a JsonPath by applying a target expr to every element of the list.
    The target expr would typically contain relative JsonPaths, which are bound to an ObjectRef, which in turn
    is populated by JsonMapper.eval(). The JsonMapper effectively creates a new scope for its target expr.

    JsonMapper is executed in two phases:
    - the first phase is handled by Expr subclass JsonMapperDispatch, which constructs one nested DataRow per source
      list element and evaluates the target expr within that (the nested DataRows are stored as a NestedRowList in the
      slot of JsonMapperDispatch)
    - JsonMapper.eval() collects the slot values of the target expr into its result list
    """

    target_expr_scope: ExprScope
    parent_mapper: Optional[JsonMapper]
    target_expr_eval_ctx: Optional[RowBuilder.EvalCtx]

    def __init__(self, src_expr: Optional[Expr], target_expr: Optional[Expr]):
        # TODO: type spec should be list[target_expr.col_type]
        super().__init__(ts.JsonType())

        dispatch = JsonMapperDispatch(src_expr, target_expr)
        self.components.append(dispatch)
        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'map({self._src_expr}, lambda R: {self._target_expr})'

    @property
    def _src_expr(self) -> Expr:
        return self.components[0].src_expr

    @property
    def _target_expr(self) -> Expr:
        return self.components[0].target_expr

    def _equals(self, _: JsonMapper) -> bool:
        return True

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        from ..exec.expr_eval.evaluators import NestedRowList

        dispatch_slot_idx = self.components[0].slot_idx
        nested_rows = data_row.vals[dispatch_slot_idx]
        if nested_rows is None:
            data_row[self.slot_idx] = None
            return
        assert isinstance(nested_rows, NestedRowList)
        # TODO: get the materialized slot idx, instead of relying on the fact that the target_expr is always at the end
        data_row[self.slot_idx] = [row.vals[-1] for row in nested_rows.rows]

    def _as_dict(self) -> dict:
        """
        We only serialize src and target exprs, everything else is re-created at runtime.
        """
        return {'components': [self._src_expr.as_dict(), self._target_expr.as_dict()]}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> JsonMapper:
        assert len(components) == 2
        src_expr, target_expr = components[0], components[1]
        return cls(src_expr, target_expr)


class JsonMapperDispatch(Expr):
    """
    An operational Expr (ie, it doesn't represent any syntactic element) that is used by JsonMapper to materialize
    its input DataRows. It has the same dependencies as the originating JsonMapper.

    - The execution (= row dispatch) is handled by an expr_eval.Evaluator (JsonMapperDispatcher).
    - It stores a NestedRowList instance in its slot.
    """

    target_expr_scope: ExprScope
    parent_mapper: Optional[JsonMapperDispatch]
    target_expr_eval_ctx: Optional[RowBuilder.EvalCtx]

    def __init__(self, src_expr: Expr, target_expr: Expr):
        super().__init__(ts.InvalidType())

        # we're creating a new scope, but we don't know yet whether this is nested within another JsonMapper;
        # this gets resolved in bind_rel_paths(); for now we assume we're in the global scope
        self.target_expr_scope = ExprScope(_GLOBAL_SCOPE)

        from .object_ref import ObjectRef

        self.components = [src_expr, target_expr]
        self.parent_mapper = None
        self.target_expr_eval_ctx = None

        # Intentionally create the id now, before adding the scope anchor; this ensures that JsonMapperDispatch
        # instances will be recognized as equal so long as they have the same src_expr and target_expr.
        # TODO: Might this cause problems after certain substitutions?
        self.id = self._create_id()

        scope_anchor = ObjectRef(self.target_expr_scope, self)
        self.components.append(scope_anchor)

    def _bind_rel_paths(self, mapper: Optional[JsonMapperDispatch] = None) -> None:
        self.src_expr._bind_rel_paths(mapper)
        self.target_expr._bind_rel_paths(self)
        self.parent_mapper = mapper
        parent_scope = _GLOBAL_SCOPE if mapper is None else mapper.target_expr_scope
        self.target_expr_scope.parent = parent_scope

    def equals(self, other: Expr) -> bool:
        """
        We override equals() because we need to avoid comparing our scope anchor.
        """
        if type(self) is not type(other):
            return False
        return self.src_expr.equals(other.src_expr) and self.target_expr.equals(other.target_expr)

    def scope(self) -> ExprScope:
        # need to ignore target_expr
        return self.src_expr.scope()

    def dependencies(self) -> list[Expr]:
        result = [self.src_expr]
        result.extend(self._target_dependencies(self.target_expr))
        return result

    def _target_dependencies(self, e: Expr) -> list[Expr]:
        """
        Return all subexprs of e of which the scope isn't contained in target_expr_scope.
        Those need to be evaluated before us.
        """
        expr_scope = e.scope()
        if not expr_scope.is_contained_in(self.target_expr_scope):
            return [e]
        result: list[Expr] = []
        for c in e.components:
            result.extend(self._target_dependencies(c))
        return result

    @property
    def src_expr(self) -> Expr:
        return self.components[0]

    @property
    def target_expr(self) -> Expr:
        return self.components[1]

    @property
    def scope_anchor(self) -> 'ObjectRef':
        from .object_ref import ObjectRef

        result = self.components[2]
        assert isinstance(result, ObjectRef)
        return result

    def __repr__(self) -> str:
        return 'JsonMapperDispatch()'

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        # eval is handled by JsonMapperDispatcher
        raise AssertionError('this should never be called')

    def _as_dict(self) -> dict:
        """
        JsonMapperDispatch instances are only created by the JsonMapper c'tor and never need to be serialized.
        """
        raise AssertionError('this should never be called')

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> JsonMapperDispatch:
        raise AssertionError('this should never be called')

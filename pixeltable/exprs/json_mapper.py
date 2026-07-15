from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any

import sqlalchemy as sql

import pixeltable.type_system as ts
from pixeltable import exceptions as excs

from .data_row import DataRow
from .expr import _GLOBAL_SCOPE, Expr, ExprScope
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache

if TYPE_CHECKING:
    from .object_ref import ObjectRef


class JsonMapper(Expr):
    """
    JsonMapper transforms the list output of a JsonPath into a new list in one of three ways:
    - MAP: every source element is replaced by the target expr evaluated on it
    - FILTER: keep source elements for which the target expr evaluates to True
    - SORT: source elements are reproduced in the order given by the target expr

    The target expr typically contains relative JsonPaths, which are bound to an ObjectRef that is populated with
    the current source element. The JsonMapper effectively creates a new scope for it.

    JsonMapper is executed in two phases:
    - the first phase is handled by Expr subclass JsonMapperDispatch, which constructs one nested DataRow per source
      list element and evaluates the target expr within that (the nested DataRows are stored as a NestedRowList in
      the slot of JsonMapperDispatch)
    - JsonMapper.eval() combines the nested rows according to op
    """

    class Operator(enum.Enum):
        MAP = 0
        FILTER = 1
        SORT = 2

    target_expr_scope: ExprScope
    parent_mapper: JsonMapper | None
    target_expr_eval_ctx: RowBuilder.EvalCtx | None
    op: Operator
    asc: bool

    def __init__(self, src_expr: Expr, target_expr: Expr, op: JsonMapper.Operator, asc: bool = True):
        # the result element is the target expr (for a map) or the source list's element (for a filter or sort);
        # when that element type isn't known, the result is an untyped list
        element_type: ts.ColumnType | None
        if op == JsonMapper.Operator.MAP:
            element_type = target_expr.col_type
        elif isinstance(src_expr.col_type, ts.JsonType):
            element_type = src_expr.col_type.array_element_type()
        else:
            element_type = None

        col_type: ts.ColumnType
        if element_type is None or (isinstance(element_type, ts.JsonType) and element_type.type_schema is None):
            col_type = ts.JsonType(nullable=True)
        else:
            col_type = ts.JsonType(ts.JsonType.TypeSchema([], variadic_type=element_type), nullable=True)
        super().__init__(col_type)

        self.op = op
        self.asc = asc
        self.components.append(JsonMapperDispatch(src_expr, target_expr, op))
        self.id = self._create_id()

    def __repr__(self) -> str:
        if self.op == JsonMapper.Operator.FILTER:
            return f'{self._src_expr}.filter(lambda R: {self._target_expr})'
        if self.op == JsonMapper.Operator.SORT:
            asc_repr = '' if self.asc else ', asc=False'
            return f'{self._src_expr}.sort(lambda R: {self._target_expr}{asc_repr})'
        return f'{self._src_expr}.map(lambda R: {self._target_expr})'

    @property
    def _src_expr(self) -> Expr:
        return self.components[0].src_expr

    @property
    def _target_expr(self) -> Expr:
        return self.components[0].target_expr

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('op', self.op.value), ('asc', self.asc)]

    def _equals(self, other: JsonMapper) -> bool:
        return self.op == other.op and self.asc == other.asc

    def sql_expr(self, _: SqlElementCache) -> sql.ColumnElement | None:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        from ..exec.expr_eval.evaluators import NestedRowList

        dispatch_slot_idx = self.components[0].slot_idx
        nested_rows = data_row.vals[dispatch_slot_idx]
        if nested_rows is None:
            data_row[self.slot_idx] = None
            return
        assert isinstance(nested_rows, NestedRowList)
        target_slot = nested_rows.target_slot_idx
        element_slot = nested_rows.element_slot_idx
        rows = nested_rows.rows
        if self.op == JsonMapper.Operator.MAP:
            data_row[self.slot_idx] = [row.vals[target_slot] for row in rows]
        elif self.op == JsonMapper.Operator.FILTER:
            data_row[self.slot_idx] = [row.vals[element_slot] for row in rows if row.vals[target_slot]]
        else:
            try:
                ordered = sorted(rows, key=lambda row: row.vals[target_slot], reverse=not self.asc)
            except TypeError as e:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    'sort(): the sort keys are not orderable (e.g. a mix of incompatible types or a null value)',
                ) from e
            data_row[self.slot_idx] = [row.vals[element_slot] for row in ordered]

    def _as_dict(self) -> dict:
        # only the src and target exprs are serialized; the dispatch is re-created at runtime
        return {
            'op': self.op.value,
            'asc': self.asc,
            'components': [self._src_expr.as_dict(), self._target_expr.as_dict()],
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr], tbl_versions: Any = None) -> JsonMapper:
        assert len(components) == 2
        # a legacy JsonMapper (metadata v55 and earlier) was always a map and has no 'op' or 'asc'
        op = cls.Operator(d['op']) if 'op' in d else cls.Operator.MAP
        return cls(components[0], components[1], op, d.get('asc', True))


class JsonMapperDispatch(Expr):
    """
    An operational Expr (ie, it doesn't represent any syntactic element) that is used by JsonMapper to materialize
    the nested DataRows in which the target expr is evaluated. It has the same dependencies as the originating
    JsonMapper.

    - The execution (= row dispatch) is handled by an expr_eval.Evaluator (JsonMapperDispatcher).
    - It stores a NestedRowList instance in its slot.
    """

    target_expr_scope: ExprScope
    parent_mapper: JsonMapperDispatch | None
    target_expr_eval_ctx: RowBuilder.EvalCtx | None
    op: JsonMapper.Operator

    def __init__(self, src_expr: Expr, target_expr: Expr, op: JsonMapper.Operator):
        super().__init__(ts.InvalidType())

        # we're creating a new scope, but we don't know yet whether this is nested within another JsonMapper;
        # this gets resolved in bind_rel_paths(); for now we assume we're in the global scope
        self.target_expr_scope = ExprScope(_GLOBAL_SCOPE)

        from .object_ref import ObjectRef

        self.op = op
        self.components = [src_expr, target_expr]
        self.parent_mapper = None
        self.target_expr_eval_ctx = None

        # Intentionally create the id now, before adding the scope anchor; this ensures that JsonMapperDispatch
        # instances will be recognized as equal so long as they have the same src, target and op.
        # TODO: Might this cause problems after certain substitutions?
        self.id = self._create_id()

        scope_anchor = ObjectRef(self.target_expr_scope, self)
        self.components.append(scope_anchor)

    def _bind_rel_paths(self, mapper: JsonMapperDispatch | None = None) -> None:
        self.src_expr._bind_rel_paths(mapper)
        self.target_expr._bind_rel_paths(self)
        self.parent_mapper = mapper
        parent_scope = _GLOBAL_SCOPE if mapper is None else mapper.target_expr_scope
        self.target_expr_scope.parent = parent_scope

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('op', self.op.value)]

    def equals(self, other: Expr) -> bool:
        """
        We override equals() because we need to avoid comparing our scope anchor.
        """
        if type(self) is not type(other):
            return False
        assert isinstance(other, JsonMapperDispatch)
        if self.op != other.op:
            return False
        return self.src_expr.equals(other.src_expr) and self.target_expr.equals(other.target_expr)

    def scope(self) -> ExprScope:
        # need to ignore target_expr
        return self.src_expr.scope()

    def dependencies(self) -> list[Expr]:
        return [self.src_expr, *self._target_dependencies(self.target_expr)]

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

        result = self.components[-1]
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
    def _from_dict(cls, d: dict, components: list[Expr], tbl_versions: Any = None) -> JsonMapperDispatch:
        raise AssertionError('this should never be called')

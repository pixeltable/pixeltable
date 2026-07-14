from __future__ import annotations

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
    JsonMapper transforms the list output of a JsonPath into a new list. Every source element is optionally
    transformed by a target expr, retained or dropped according to a filter expr, and ordered by a key expr:
    - with a target expr and no filter expr, this is a map: every element is transformed
    - with a filter expr and no target expr, this is a filter: only the elements for which the filter expr is
      truthy are retained, unchanged
    - with a key expr, this is a sort: the elements are reproduced in the order given by the key expr (ascending
      when asc is True, descending otherwise)
    - these can be combined, and all can be omitted (in which case the source list is reproduced as-is)

    The target, filter and key exprs typically contain relative JsonPaths, which are bound to an ObjectRef that is
    populated with the current source element. The JsonMapper effectively creates a new scope for them.

    JsonMapper is executed in two phases:
    - the first phase is handled by Expr subclass JsonMapperDispatch, which constructs one nested DataRow per source
      list element and evaluates the target, filter and key exprs within that (the nested DataRows are stored as a
      NestedRowList in the slot of JsonMapperDispatch)
    - JsonMapper.eval() collects the per-element value of every nested row that passes the filter expr into its
      result list, ordered by the key expr when there is one
    """

    target_expr_scope: ExprScope
    parent_mapper: JsonMapper | None
    target_expr_eval_ctx: RowBuilder.EvalCtx | None
    asc: bool

    def __init__(
        self,
        src_expr: Expr | None,
        target_expr: Expr | None,
        filter_expr: Expr | None = None,
        key_expr: Expr | None = None,
        asc: bool = True,
    ):
        # the result is a list whose element is the target expr (for a map), or the source list's element (for a
        # filter, sort or plain reproduction); when that element type isn't known, the result is an untyped list
        if target_expr is not None:
            element_type: ts.ColumnType | None = target_expr.col_type
        elif src_expr is not None and isinstance(src_expr.col_type, ts.JsonType):
            element_type = src_expr.col_type.array_element_type()
        else:
            element_type = None
        if element_type is None or (isinstance(element_type, ts.JsonType) and element_type.type_schema is None):
            col_type: ts.ColumnType = ts.JsonType(nullable=True)
        else:
            col_type = ts.JsonType(ts.JsonType.TypeSchema([], variadic_type=element_type), nullable=True)
        super().__init__(col_type)

        self.asc = asc
        dispatch = JsonMapperDispatch(src_expr, target_expr, filter_expr, key_expr)
        self.components.append(dispatch)
        self.id = self._create_id()

    def __repr__(self) -> str:
        if self._key_expr is not None:
            asc_repr = '' if self.asc else ', asc=False'
            return f'{self._src_expr}.sort(lambda R: {self._key_expr}{asc_repr})'
        if self._target_expr is None:
            return f'{self._src_expr}.filter(lambda R: {self._filter_expr})'
        return f'{self._src_expr}.map(lambda R: {self._target_expr})'

    @property
    def _src_expr(self) -> Expr:
        return self.components[0].src_expr

    @property
    def _target_expr(self) -> Expr | None:
        return self.components[0].target_expr

    @property
    def _filter_expr(self) -> Expr | None:
        return self.components[0].filter_expr

    @property
    def _key_expr(self) -> Expr | None:
        return self.components[0].key_expr

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('asc', self.asc)]

    def _equals(self, other: JsonMapper) -> bool:
        return self.asc == other.asc

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
        value_slot = nested_rows.value_slot_idx
        filter_slot = nested_rows.filter_slot_idx
        key_slot = nested_rows.key_slot_idx
        rows = [row for row in nested_rows.rows if filter_slot is None or row.vals[filter_slot]]
        if key_slot is not None:
            try:
                rows = sorted(rows, key=lambda row: row.vals[key_slot], reverse=not self.asc)
            except TypeError as e:
                raise excs.RequestError(
                    excs.ErrorCode.INVALID_ARGUMENT,
                    'sort(): the sort keys are not orderable (e.g. a mix of incompatible types or a null value)',
                ) from e
        data_row[self.slot_idx] = [row.vals[value_slot] for row in rows]

    def _as_dict(self) -> dict:
        """
        We only serialize src, target and filter exprs, everything else is re-created at runtime.
        The components are src_expr, followed by target_expr (if present), followed by filter_expr (if present).
        """
        components = [self._src_expr.as_dict()]
        if self._target_expr is not None:
            components.append(self._target_expr.as_dict())
        if self._filter_expr is not None:
            components.append(self._filter_expr.as_dict())
        if self._key_expr is not None:
            components.append(self._key_expr.as_dict())
        return {
            'has_target': self._target_expr is not None,
            'has_filter': self._filter_expr is not None,
            'has_key': self._key_expr is not None,
            'asc': self.asc,
            'components': components,
        }

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr], tbl_versions: Any = None) -> JsonMapper:
        src_expr = components[0]
        idx = 1
        target_expr: Expr | None = None
        if d['has_target']:
            target_expr = components[idx]
            idx += 1
        filter_expr: Expr | None = None
        if d['has_filter']:
            filter_expr = components[idx]
            idx += 1
        key_expr: Expr | None = None
        if d['has_key']:
            key_expr = components[idx]
        return cls(src_expr, target_expr, filter_expr, key_expr, d['asc'])


class JsonMapperDispatch(Expr):
    """
    An operational Expr (ie, it doesn't represent any syntactic element) that is used by JsonMapper to materialize
    the nested DataRows in which the target, filter and key exprs are evaluated. It has the same dependencies as the
    originating JsonMapper.

    - The execution (= row dispatch) is handled by an expr_eval.Evaluator (JsonMapperDispatcher).
    - It stores a NestedRowList instance in its slot.
    """

    target_expr_scope: ExprScope
    parent_mapper: JsonMapperDispatch | None
    target_expr_eval_ctx: RowBuilder.EvalCtx | None
    has_target: bool
    has_filter: bool
    has_key: bool

    def __init__(self, src_expr: Expr, target_expr: Expr | None, filter_expr: Expr | None, key_expr: Expr | None):
        super().__init__(ts.InvalidType())

        # we're creating a new scope, but we don't know yet whether this is nested within another JsonMapper;
        # this gets resolved in bind_rel_paths(); for now we assume we're in the global scope
        self.target_expr_scope = ExprScope(_GLOBAL_SCOPE)

        from .object_ref import ObjectRef

        self.has_target = target_expr is not None
        self.has_filter = filter_expr is not None
        self.has_key = key_expr is not None
        self.components = [src_expr]
        if target_expr is not None:
            self.components.append(target_expr)
        if filter_expr is not None:
            self.components.append(filter_expr)
        if key_expr is not None:
            self.components.append(key_expr)
        self.parent_mapper = None
        self.target_expr_eval_ctx = None

        # Intentionally create the id now, before adding the scope anchor; this ensures that JsonMapperDispatch
        # instances will be recognized as equal so long as they have the same src, target, filter and key exprs.
        # TODO: Might this cause problems after certain substitutions?
        self.id = self._create_id()

        scope_anchor = ObjectRef(self.target_expr_scope, self)
        self.components.append(scope_anchor)

    def _bind_rel_paths(self, mapper: JsonMapperDispatch | None = None) -> None:
        self.src_expr._bind_rel_paths(mapper)
        if self.target_expr is not None:
            self.target_expr._bind_rel_paths(self)
        if self.filter_expr is not None:
            self.filter_expr._bind_rel_paths(self)
        if self.key_expr is not None:
            self.key_expr._bind_rel_paths(self)
        self.parent_mapper = mapper
        parent_scope = _GLOBAL_SCOPE if mapper is None else mapper.target_expr_scope
        self.target_expr_scope.parent = parent_scope

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [
            *super()._id_attrs(),
            ('has_target', self.has_target),
            ('has_filter', self.has_filter),
            ('has_key', self.has_key),
        ]

    def equals(self, other: Expr) -> bool:
        """
        We override equals() because we need to avoid comparing our scope anchor.
        """
        if type(self) is not type(other):
            return False
        assert isinstance(other, JsonMapperDispatch)
        if self.has_target != other.has_target or self.has_filter != other.has_filter or self.has_key != other.has_key:
            return False
        if not self.src_expr.equals(other.src_expr):
            return False
        if self.target_expr is not None and not self.target_expr.equals(other.target_expr):
            return False
        if self.filter_expr is not None and not self.filter_expr.equals(other.filter_expr):
            return False
        return self.key_expr is None or self.key_expr.equals(other.key_expr)

    def scope(self) -> ExprScope:
        # need to ignore target_expr, filter_expr and key_expr
        return self.src_expr.scope()

    def dependencies(self) -> list[Expr]:
        result = [self.src_expr]
        scoped_exprs = [e for e in (self.target_expr, self.filter_expr, self.key_expr) if e is not None]
        for e in scoped_exprs:
            result.extend(self._target_dependencies(e))
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
    def target_expr(self) -> Expr | None:
        return self.components[1] if self.has_target else None

    @property
    def filter_expr(self) -> Expr | None:
        if not self.has_filter:
            return None
        return self.components[2] if self.has_target else self.components[1]

    @property
    def key_expr(self) -> Expr | None:
        if not self.has_key:
            return None
        return self.components[1 + int(self.has_target) + int(self.has_filter)]

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

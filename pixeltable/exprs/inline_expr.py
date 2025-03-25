from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.
    """

    def __init__(self, elements: Iterable):
        exprs = []
        for el in elements:
            if isinstance(el, Expr):
                exprs.append(el)
            elif isinstance(el, (list, tuple)):
                exprs.append(Expr.from_array(el))
            else:
                exprs.append(Literal(el))

        inferred_element_type: Optional[ts.ColumnType] = ts.InvalidType()
        for i, expr in enumerate(exprs):
            supertype = inferred_element_type.supertype(expr.col_type)
            if supertype is None:
                raise excs.Error(
                    f'Could not infer element type of array: element of type `{expr.col_type}` at index {i} '
                    f'is not compatible with type `{inferred_element_type}` of preceding elements'
                )
            inferred_element_type = supertype

        if inferred_element_type.is_scalar_type():
            col_type = ts.ArrayType((len(exprs),), inferred_element_type)
        elif inferred_element_type.is_array_type():
            assert isinstance(inferred_element_type, ts.ArrayType)
            col_type = ts.ArrayType(
                (len(exprs), *inferred_element_type.shape), ts.ColumnType.make_type(inferred_element_type.dtype)
            )
        else:
            raise excs.Error(f'Element type is not a valid dtype for an array: {inferred_element_type}')

        super().__init__(col_type)
        self.components.extend(exprs)
        self.id = self._create_id()

    def __repr__(self) -> str:
        elem_strs = [str(expr) for expr in self.components]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, _: InlineArray) -> bool:
        return True  # Always true if components match

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = np.array([data_row[el.slot_idx] for el in self.components])

    def _as_dict(self) -> dict:
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, _: dict, components: list[Expr]) -> InlineArray:
        try:
            return cls(components)
        except excs.Error:
            # For legacy compatibility reasons, we need to try constructing as an `InlineList`.
            # This is because in schema versions <= 19, `InlineArray` was serialized incorrectly, and
            # there is no way to determine the correct expression type until the subexpressions are
            # loaded and their types are known.
            return InlineList(components)  # type: ignore[return-value]

    def as_literal(self) -> Optional[Literal]:
        assert isinstance(self.col_type, ts.ArrayType)
        if not all(isinstance(comp, Literal) for comp in self.components):
            return None
        return Literal(
            np.array([c.as_literal().val for c in self.components], dtype=self.col_type.numpy_dtype()), self.col_type
        )


class InlineList(Expr):
    """
    List 'literal' which can use Exprs as values.
    """

    def __init__(self, elements: Iterable):
        exprs = [Expr.from_object(el) for el in elements]

        json_schema = {
            'type': 'array',
            'prefixItems': [expr.col_type.to_json_schema() for expr in exprs],
            'items': False,  # No additional items (fixed length)
        }

        super().__init__(ts.JsonType(json_schema))
        self.components.extend(exprs)
        self.id = self._create_id()

    def __repr__(self) -> str:
        elem_strs = [str(expr) for expr in self.components]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, _: InlineList) -> bool:
        return True  # Always true if components match

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, _: RowBuilder) -> None:
        data_row[self.slot_idx] = [data_row[el.slot_idx] for el in self.components]

    def _as_dict(self) -> dict:
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, _: dict, components: list[Expr]) -> InlineList:
        return cls(components)

    def as_literal(self) -> Optional[Literal]:
        if not all(isinstance(comp, Literal) for comp in self.components):
            return None
        return Literal([c.as_literal().val for c in self.components], self.col_type)


class InlineDict(Expr):
    """
    Dictionary 'literal' which can use Exprs as values.
    """

    keys: list[str]

    def __init__(self, d: dict[str, Any]):
        self.keys = []
        exprs: list[Expr] = []
        for key, val in d.items():
            if not isinstance(key, str):
                raise excs.Error(f'Dictionary requires string keys; {key} has type {type(key)}')
            self.keys.append(key)
            exprs.append(Expr.from_object(val))

        json_schema: Optional[dict[str, Any]]
        try:
            json_schema = {
                'type': 'object',
                'properties': {key: expr.col_type.to_json_schema() for key, expr in zip(self.keys, exprs)},
            }
        except excs.Error:
            # InlineDicts are used to store iterator arguments, which are not required to be valid JSON types,
            # so we can't always construct a valid schema.
            json_schema = None

        super().__init__(ts.JsonType(json_schema))
        self.components.extend(exprs)
        self.id = self._create_id()

    def __repr__(self) -> str:
        item_strs = [f"'{key}': {expr}" for key, expr in zip(self.keys, self.components)]
        return '{' + ', '.join(item_strs) + '}'

    def _equals(self, other: InlineDict) -> bool:
        # The dict values are just the components, which have already been checked
        return self.keys == other.keys

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('keys', self.keys)]

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, _: RowBuilder) -> None:
        assert len(self.keys) == len(self.components)
        data_row[self.slot_idx] = {key: data_row[expr.slot_idx] for key, expr in zip(self.keys, self.components)}

    def to_kwargs(self) -> dict[str, Any]:
        """Deconstructs this expression into a dictionary by recursively unwrapping all Literals,
        InlineDicts, and InlineLists."""
        return InlineDict._to_kwarg_element(self)

    @classmethod
    def _to_kwarg_element(cls, expr: Expr) -> Any:
        if isinstance(expr, Literal):
            return expr.val
        if isinstance(expr, InlineDict):
            return {key: cls._to_kwarg_element(val) for key, val in zip(expr.keys, expr.components)}
        if isinstance(expr, InlineList):
            return [cls._to_kwarg_element(el) for el in expr.components]
        return expr

    def _as_dict(self) -> dict[str, Any]:
        return {'keys': self.keys, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> InlineDict:
        assert 'keys' in d
        assert len(d['keys']) == len(components)
        arg = dict(zip(d['keys'], components))
        return InlineDict(arg)

    def as_literal(self) -> Optional[Literal]:
        if not all(isinstance(comp, Literal) for comp in self.components):
            return None
        return Literal(dict(zip(self.keys, (c.as_literal().val for c in self.components))), self.col_type)

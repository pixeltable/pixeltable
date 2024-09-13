from __future__ import annotations

import copy
from typing import Any, Iterable, Optional

import numpy as np
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .literal import Literal
from .row_builder import RowBuilder


class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.
    """

    elements: list[Expr]

    def __init__(self, elements: Iterable):
        self.elements = []
        for el in elements:
            if isinstance(el, Expr):
                self.elements.append(el)
            elif isinstance(el, list) or isinstance(el, tuple):
                self.elements.append(InlineArray(el))
            else:
                self.elements.append(Literal(el))

        inferred_element_type: Optional[ts.ColumnType] = ts.InvalidType()
        for expr in self.elements:
            inferred_element_type = inferred_element_type.supertype(expr.col_type)
            if inferred_element_type is None:
                raise excs.Error('Could not infer element type of array')

        if inferred_element_type.is_scalar_type():
            col_type = ts.ArrayType((len(self.elements),), inferred_element_type)
        elif inferred_element_type.is_array_type():
            assert isinstance(inferred_element_type, ts.ArrayType)
            col_type = ts.ArrayType(
                (len(self.elements), *inferred_element_type.shape),
                ts.ColumnType.make_type(inferred_element_type.dtype)
            )
        else:
            raise excs.Error(f'Element type is not a valid dtype for an array: {inferred_element_type}')

        super().__init__(col_type)
        self.components.extend(self.elements)
        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(expr) for expr in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, _: InlineList) -> bool:
        return True  # Always true if components match

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [('elements', self.elements)]

    def sql_expr(self) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        data_row[self.slot_idx] = np.array([data_row[el.slot_idx] for el in self.components])

    def _as_dict(self) -> dict:
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, _: dict, components: list[Expr]) -> Expr:
        try:
            return cls(components)
        except excs.Error:
            # For legacy compatibility reasons, we need to try constructing as an `InlineList`.
            # This is because in schema versions <= 19, `InlineArray` was serialized incorrectly, and
            # there is no way to determine the correct expression type until the subexpressions are
            # loaded and their types are known.
            return InlineList(components)


class InlineList(Expr):
    """
    List 'literal' which can use Exprs as values.
    """

    elements: list[Expr]

    def __init__(self, elements: Iterable):
        self.elements = []
        for el in elements:
            if isinstance(el, Expr):
                self.elements.append(el)
            elif isinstance(el, list) or isinstance(el, tuple):
                self.elements.append(InlineList(el))
            elif isinstance(el, dict):
                self.elements.append(InlineDict(el))
            else:
                self.elements.append(Literal(el))

        super().__init__(ts.JsonType())
        self.components.extend(self.elements)
        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(expr) for expr in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, _: InlineList) -> bool:
        return True  # Always true if components match

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [('elements', self.elements)]

    def sql_expr(self) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, _: RowBuilder) -> None:
        data_row[self.slot_idx] = [data_row[el.slot_idx] for el in self.components]

    def _as_dict(self) -> dict:
        return super()._as_dict()

    @classmethod
    def _from_dict(cls, _: dict, components: list[Expr]) -> Expr:
        return cls(components)


class InlineDict(Expr):
    """
    Dictionary 'literal' which can use Exprs as values.
    """

    expr_dict: dict[str, Expr]

    def __init__(self, d: dict[str, Any]):
        self.expr_dict = {}
        for key, val in d.items():
            if not isinstance(key, str):
                raise excs.Error(f'Dictionary requires string keys; {key} has type {type(key)}')
            if isinstance(val, Expr):
                self.expr_dict[key] = val
            elif isinstance(val, dict):
                self.expr_dict[key] = InlineDict(val)
            elif isinstance(val, list) or isinstance(val, tuple):
                self.expr_dict[key] = InlineList(val)
            else:
                self.expr_dict[key] = Literal(val)

        super().__init__(ts.JsonType())
        self.components.extend(self.expr_dict.values())
        self.id = self._create_id()

    def __str__(self) -> str:
        item_strs = list(f"'{key}': {str(expr)}" for key, expr in self.expr_dict.items())
        return '{' + ', '.join(item_strs) + '}'

    def _equals(self, other: InlineDict) -> bool:
        # The values are just the components, which have already been checked
        return list(self.expr_dict.keys()) == list(other.expr_dict.keys())

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [('expr_dict', self.expr_dict)]

    def sql_expr(self) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, _: RowBuilder) -> None:
        assert len(self.expr_dict) == len(self.components)
        data_row[self.slot_idx] = {
            key: data_row[self.components[i].slot_idx]
            for i, key in enumerate(self.expr_dict.keys())
        }

    def unwrap(self) -> dict[str, Any]:
        """Deconstructs this expression into a dictionary by unwrapping all Literals,
        InlineDicts, and InlineLists."""
        return InlineDict._to_dict_element(self)

    @classmethod
    def _to_dict_element(cls, expr: Expr) -> Any:
        if isinstance(expr, Literal):
            return expr.val
        if isinstance(expr, InlineDict):
            return {key: cls._to_dict_element(val) for key, val in expr.expr_dict.items()}
        if isinstance(expr, InlineList):
            return [cls._to_dict_element(el) for el in expr.elements]
        return expr

    def _as_dict(self) -> dict[str, Any]:
        return {'keys': list(self.expr_dict.keys()), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Expr:
        assert 'keys' in d
        assert len(d['keys']) == len(components)
        arg = dict(zip(d['keys'], components))
        return InlineDict(arg)

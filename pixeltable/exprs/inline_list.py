from __future__ import annotations

import copy
from typing import Iterable, Optional, List, Any, Dict, Tuple

import numpy as np
import sqlalchemy as sql

import pixeltable.type_system as ts
from .data_row import DataRow
from .expr import Expr
from .inline_dict import InlineDict
from .row_builder import RowBuilder


class InlineList(Expr):
    """
    Array 'literal' which can use Exprs as values.

    The literal can be cast as either a pixeltable `ArrayType` or `JsonType`. If `force_json`
    is `True`, it will always be cast as a `JsonType`. If `force_json` is `False`, it will be cast as an
    `ArrayType` if it is a homogenous array of scalars or arrays, or a `JsonType` otherwise.
    """

    elements: list[tuple[Optional[int], Any]]

    def __init__(self, elements: Iterable):
        # we need to call this in order to populate self.components
        super().__init__(ts.JsonType())

        # self.elements contains
        # - for Expr elements: (index into components, None)
        # - for non-Expr elements: (None, value)
        self.elements = []
        for el in elements:
            if isinstance(el, list):
                el = InlineList(el)
            elif isinstance(el, dict):
                el = InlineDict(el)

            if isinstance(el, Expr):
                self.elements.append((len(self.components), None))
                self.components.append(el)
            else:
                self.elements.append((None, el))

        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(val) if val is not None else str(self.components[idx]) for idx, val in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, other: InlineList) -> bool:
        return self.elements == other.elements

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return super()._id_attrs() + [('elements', self.elements)]

    def sql_expr(self) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = [None] * len(self.elements)
        for i, (child_idx, val) in enumerate(self.elements):
            if child_idx is not None:
                result[i] = data_row[self.components[child_idx].slot_idx]
            else:
                result[i] = copy.deepcopy(val)
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, 'is_json': self.col_type.is_json_type(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Expr:
        assert 'elements' in d
        arg: list[Any] = []
        for idx, val in d['elements']:
            # TODO Normalize idx -1 to None via schema migrations.
            # Long-term we should not be allowing idx == -1.
            if idx is not None and idx >= 0:  # Older schemas might have -1 instead of None
                arg.append(components[idx])
            else:
                arg.append(val)
        return cls(tuple(arg))

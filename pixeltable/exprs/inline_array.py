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


class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.

    The literal can be cast as either a pixeltable `ArrayType` or `JsonType`. If `force_json`
    is `True`, it will always be cast as a `JsonType`. If `force_json` is `False`, it will be cast as an
    `ArrayType` if it is a homogenous array of scalars or arrays, or a `JsonType` otherwise.
    """

    elements: List[Tuple[Optional[int], Any]]

    def __init__(self, elements: Iterable, force_json: bool = False):
        # we need to call this in order to populate self.components
        super().__init__(ts.ArrayType((len(elements),), ts.IntType()))

        # elements contains
        # - for Expr elements: (index into components, None)
        # - for non-Expr elements: (None, value)
        self.elements = []
        for el in elements:
            el = copy.deepcopy(el)
            if isinstance(el, list):
                # If col_type is an ArrayType, we'll require it to be a multidimensional array
                # of the specified underlying type
                el = InlineArray(tuple(el), force_json)
            if isinstance(el, dict):
                el = InlineDict(el)
            if isinstance(el, Expr):
                self.elements.append((len(self.components), None))
                self.components.append(el)
            else:
                self.elements.append((None, el))

        inferred_element_type: Optional[ts.ColumnType] = ts.InvalidType()
        if not force_json:
            # try to infer the element type
            for idx, val in self.elements:
                if idx is not None:
                    inferred_element_type = inferred_element_type.supertype(self.components[idx].col_type)
                else:
                    inferred_element_type = inferred_element_type.supertype(ts.ColumnType.infer_literal_type(val))
                if inferred_element_type is None:
                    break

        if force_json or inferred_element_type is None:
            # JSON conversion is forced, or there is no common supertype
            # TODO: make sure this doesn't contain Images
            self.col_type = ts.JsonType()
        elif inferred_element_type.is_scalar_type():
            self.col_type = ts.ArrayType((len(self.elements),), inferred_element_type)
        elif inferred_element_type.is_array_type():
            assert isinstance(inferred_element_type, ts.ArrayType)
            self.col_type = ts.ArrayType(
                (len(self.elements), *inferred_element_type.shape), ts.ColumnType.make_type(inferred_element_type.dtype))
        else:
            self.col_type = ts.JsonType()

        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(val) if val is not None else str(self.components[idx]) for idx, val in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, other: InlineDict) -> bool:
        return self.elements == other.elements

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('elements', self.elements)]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = [None] * len(self.elements)
        for i, (child_idx, val) in enumerate(self.elements):
            if child_idx is not None:
                result[i] = data_row[self.components[child_idx].slot_idx]
            else:
                result[i] = copy.deepcopy(val)
        if self.col_type.is_array_type():
            data_row[self.slot_idx] = np.array(result)
        else:
            data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, 'is_json': self.col_type.is_json_type(), **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'elements' in d
        arg: List[Any] = []
        for idx, val in d['elements']:
            # TODO Normalize idx -1 to None via schema migrations.
            # Long-term we should not be allowing idx == -1.
            if idx is not None and idx >= 0:  # Older schemas might have -1 instead of None
                arg.append(components[idx])
            else:
                arg.append(val)
        # in order to avoid a schema version change, we'll interpret the absence of 'is_json' to indicate an ArrayType
        is_json = d.get('is_json', False)
        return cls(tuple(arg), force_json=is_json)

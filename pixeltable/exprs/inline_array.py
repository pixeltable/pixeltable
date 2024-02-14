from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple
import copy

import sqlalchemy as sql
import numpy as np

from .expr import Expr
from .data_row import DataRow
from .inline_dict import InlineDict
from .row_builder import RowBuilder
import pixeltable.catalog as catalog
import pixeltable.type_system as ts


class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.

    The literal can be cast as either a pixeltable `ArrayType` or `JsonType`. If `force_json`
    is `True`, it will always be cast as a `JsonType`. If `force_json` is `False`, it will be cast as an
    `ArrayType` if it is a homogenous array of scalars or arrays, or a `JsonType` otherwise.
    """
    def __init__(self, elements: Tuple, force_json: bool = False):
        # we need to call this in order to populate self.components
        super().__init__(ts.ArrayType((len(elements),), ts.IntType()))

        # elements contains
        # - for Expr elements: (index into components, None)
        # - for non-Expr elements: (-1, value)
        self.elements: List[Tuple[int, Any]] = []
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
                self.elements.append((-1, el))

        inferred_element_type = ts.InvalidType()
        for idx, val in self.elements:
            if idx >= 0:
                inferred_element_type = ts.ColumnType.supertype(inferred_element_type, self.components[idx].col_type)
            else:
                inferred_element_type = ts.ColumnType.supertype(inferred_element_type, ts.ColumnType.infer_literal_type(val))
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
            if child_idx >= 0:
                result[i] = data_row[self.components[child_idx].slot_idx]
            else:
                result[i] = copy.deepcopy(val)
        if self.col_type.is_array_type():
            data_row[self.slot_idx] = np.array(result)
        else:
            data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'elements' in d
        arg: List[Any] = []
        for idx, val in d['elements']:
            if idx >= 0:
                arg.append(components[idx])
            else:
                arg.append(val)
        return cls(tuple(arg))

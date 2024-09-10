from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
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
        return {'elements': self.elements, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> Expr:
        assert 'elements' in d
        arg: list[Any] = []
        for idx, val in d['elements']:
            assert idx != -1
            if idx is not None:
                arg.append(components[idx])
            else:
                arg.append(val)
        return cls(tuple(arg))

class InlineDict(Expr):
    """
    Dictionary 'literal' which can use Exprs as values.
    """
    def __init__(self, d: Dict):
        super().__init__(ts.JsonType())  # we need to call this in order to populate self.components
        # dict_items contains
        # - for Expr fields: (key, index into components, None)
        # - for non-Expr fields: (key, None, value)
        self.dict_items: List[Tuple[str, Optional[int], Any]] = []
        for key, val in d.items():
            if not isinstance(key, str):
                raise excs.Error(f'Dictionary requires string keys, {key} has type {type(key)}')
            val = copy.deepcopy(val)
            if isinstance(val, dict):
                val = InlineDict(val)
            if isinstance(val, list) or isinstance(val, tuple):
                val = InlineList(val)
            if isinstance(val, Expr):
                self.dict_items.append((key, len(self.components), None))
                self.components.append(val)
            else:
                self.dict_items.append((key, None, val))

        self.type_spec: Optional[Dict[str, ts.ColumnType]] = {}
        for key, idx, _ in self.dict_items:
            if idx is None:
                # TODO: implement type inference for values
                self.type_spec = None
                break
            self.type_spec[key] = self.components[idx].col_type
        self.col_type = ts.JsonType(self.type_spec)

        self.id = self._create_id()

    def __str__(self) -> str:
        item_strs: List[str] = []
        i = 0
        def print_val(val: Any) -> str:
            if isinstance(val, str):
                return f"'{val}'"
            return str(val)
        for key, idx, val in self.dict_items:
            if idx is not None:
                item_strs.append(f"'{key}': {str(self.components[i])}")
                i += 1
            else:
                item_strs.append(f"'{key}': {print_val(val)}")
        return '{' + ', '.join(item_strs) + '}'

    def _equals(self, other: InlineDict) -> bool:
        return self.dict_items == other.dict_items

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('dict_items', self.dict_items)]

    def to_dict(self) -> Dict[str, Any]:
        """Return the original dict used to construct this"""
        return {key: val if idx is None else self.components[idx] for key, idx, val in self.dict_items}

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        result = {}
        for key, idx, val in self.dict_items:
            assert isinstance(key, str)
            if idx is not None:
                result[key] = data_row[self.components[idx].slot_idx]
            else:
                result[key] = copy.deepcopy(val)
        data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'dict_items': self.dict_items, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'dict_items' in d
        arg: dict[str, Any] = {}
        for key, idx, val in d['dict_items']:
            assert idx != -1
            if idx is not None:
                arg[key] = components[idx]
            else:
                arg[key] = val
        return cls(arg)

class InlineArray(Expr):
    """
    Array 'literal' which can use Exprs as values.

    The literal can be cast as either a pixeltable `ArrayType` or `JsonType`. If `force_json`
    is `True`, it will always be cast as a `JsonType`. If `force_json` is `False`, it will be cast as an
    `ArrayType` if it is a homogenous array of scalars or arrays, or a `JsonType` otherwise.
    """

    elements: list[tuple[Optional[int], Any]]

    def __init__(self, elements: Iterable):
        # we need to call this in order to populate self.components
        super().__init__(ts.ArrayType((len(elements),), ts.IntType()))

        # elements contains
        # - for Expr elements: (index into components, None)
        # - for non-Expr elements: (None, value)
        self.elements = []
        for el in elements:
            if isinstance(el, list):
                el = InlineArray(el)
            if isinstance(el, Expr):
                self.elements.append((len(self.components), None))
                self.components.append(el)
            else:
                self.elements.append((None, el))

        inferred_element_type: Optional[ts.ColumnType] = ts.InvalidType()
        for idx, val in self.elements:
            if idx is not None:
                inferred_element_type = inferred_element_type.supertype(self.components[idx].col_type)
            else:
                inferred_element_type = inferred_element_type.supertype(ts.ColumnType.infer_literal_type(val))
            if inferred_element_type is None:
                raise excs.Error('Could not infer element type of array')

        if inferred_element_type.is_scalar_type():
            self.col_type = ts.ArrayType((len(self.elements),), inferred_element_type)
        elif inferred_element_type.is_array_type():
            assert isinstance(inferred_element_type, ts.ArrayType)
            self.col_type = ts.ArrayType(
                (len(self.elements), *inferred_element_type.shape), ts.ColumnType.make_type(inferred_element_type.dtype))
        else:
            raise excs.Error(f'Element type is not a valid dtype for an array: {inferred_element_type}')

        self.id = self._create_id()

    def __str__(self) -> str:
        elem_strs = [str(val) if val is not None else str(self.components[idx]) for idx, val in self.elements]
        return f'[{", ".join(elem_strs)}]'

    def _equals(self, other: InlineArray) -> bool:
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
        if self.col_type.is_array_type():
            data_row[self.slot_idx] = np.array(result)
        else:
            data_row[self.slot_idx] = result

    def _as_dict(self) -> Dict:
        return {'elements': self.elements, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'elements' in d
        arg: list[Any] = []
        for idx, val in d['elements']:
            assert idx != -1
            if idx is not None:
                arg.append(components[idx])
            else:
                arg.append(val)
        return cls(tuple(arg))

from __future__ import annotations

from typing import Any, Optional, Union

import sqlalchemy as sql

from .data_row import DataRow
from .expr import Expr
from .globals import print_slice
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class ArraySlice(Expr):
    """
    Slice operation on an array, eg, t.array_col[:, 1:2].
    """

    def __init__(self, arr: Expr, index: tuple[Union[int, slice], ...]):
        assert arr.col_type.is_array_type()
        # determine result type
        super().__init__(arr.col_type)
        self.components = [arr]
        self.index = index
        self.id = self._create_id()

    def __repr__(self) -> str:
        index_strs: list[str] = []
        for el in self.index:
            if isinstance(el, int):
                index_strs.append(str(el))
            if isinstance(el, slice):
                index_strs.append(print_slice(el))
        return f'{self._array}[{", ".join(index_strs)}]'

    @property
    def _array(self) -> Expr:
        return self.components[0]

    def _equals(self, other: ArraySlice) -> bool:
        return self.index == other.index

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('index', self.index)]

    def sql_expr(self, _: SqlElementCache) -> Optional[sql.ColumnElement]:
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        val = data_row[self._array.slot_idx]
        data_row[self.slot_idx] = val[self.index]

    def _as_dict(self) -> dict:
        index: list[Any] = []
        for el in self.index:
            if isinstance(el, slice):
                index.append([el.start, el.stop, el.step])
            else:
                index.append(el)
        return {'index': index, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> ArraySlice:
        assert 'index' in d
        index = []
        for el in d['index']:
            if isinstance(el, list):
                index.append(slice(el[0], el[1], el[2]))
            else:
                index.append(el)
        return cls(components[0], tuple(index))

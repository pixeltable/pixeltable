from __future__ import annotations

from typing import Any, Iterable, Optional

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class InPredicate(Expr):
    """Predicate corresponding to the SQL IN operator."""

    def __init__(self, lhs: Expr, value_set_literal: Optional[Iterable] = None, value_set_expr: Optional[Expr] = None):
        assert (value_set_literal is None) != (value_set_expr is None)
        if not lhs.col_type.is_scalar_type():
            raise excs.Error(f'isin(): only supported for scalar types, not {lhs.col_type}')
        super().__init__(ts.BoolType())

        self.value_list: Optional[list] = None  # only contains values of the correct type
        if value_set_expr is not None:
            if not value_set_expr.col_type.is_json_type():
                raise excs.Error(
                    f'isin(): argument must have a JSON type, but {value_set_expr} has type {value_set_expr.col_type}'
                )
            self.components = [lhs.copy(), value_set_expr.copy()]
        else:
            assert value_set_literal is not None
            self.components = [lhs.copy()]
            self.value_list = self._normalize_value_set(value_set_literal)

        self.id = self._create_id()

    @property
    def _lhs(self) -> Expr:
        return self.components[0]

    @property
    def _value_set_expr(self) -> Expr:
        assert len(self.components) == 2
        return self.components[1]

    def _normalize_value_set(self, value_set: Iterable, filter_type_mismatches: bool = True) -> list:
        if not isinstance(value_set, Iterable):
            raise excs.Error(f'isin(): argument must be an Iterable (eg, list, dict, ...), not {value_set!r}')
        value_list = list(value_set)
        if not filter_type_mismatches:
            return value_list

        # ignore elements of the wrong type
        result = []
        for val in value_list:
            try:
                self._lhs.col_type.validate_literal(val)
                result.append(val)
            except TypeError:
                pass
        return result

    def __repr__(self) -> str:
        if self.value_list is not None:
            return f'{self.components[0]}.isin({self.value_list})'
        return f'{self.components[0]}.isin({self.components[1]})'

    def _equals(self, other: InPredicate) -> bool:
        return self.value_list == other.value_list

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('value_list', self.value_list)]

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        lhs_sql_exprs = sql_elements.get(self.components[0])
        if lhs_sql_exprs is None or self.value_list is None:
            return None
        return lhs_sql_exprs.in_(self.value_list)

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        lhs_val = data_row[self._lhs.slot_idx]
        if self.value_list is not None:
            data_row[self.slot_idx] = lhs_val in self.value_list
        else:
            value_set = data_row[self._value_set_expr.slot_idx]
            value_list = self._normalize_value_set(value_set, filter_type_mismatches=False)
            data_row[self.slot_idx] = lhs_val in value_list

    def _as_dict(self) -> dict:
        return {'value_list': self.value_list, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> InPredicate:
        assert 'value_list' in d
        assert len(components) <= 2
        return cls(components[0], d['value_list'], components[1] if len(components) == 2 else None)

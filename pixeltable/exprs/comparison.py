from __future__ import annotations

from typing import Any

import sqlalchemy as sql

import pixeltable.type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .globals import ComparisonOperator
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache
from .variable import Variable


class Comparison(Expr):
    is_search_arg_comparison: bool
    operator: ComparisonOperator

    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__(ts.BoolType())
        self.operator = operator

        # if this is a comparison of a column to a constant (ie, could be used as a search argument in an index lookup),
        # normalize it to <column> <operator> <constant>.
        if isinstance(op1, ColumnRef) and isinstance(op2, (Literal, Variable)):
            self.is_search_arg_comparison = True
            self.components = [op1, op2]
        elif isinstance(op1, (Literal, Variable)) and isinstance(op2, ColumnRef):
            self.is_search_arg_comparison = True
            self.components = [op2, op1]
            self.operator = self.operator.reverse()
        else:
            self.is_search_arg_comparison = False
            self.components = [op1, op2]

        self.id = self._create_id()

    def __repr__(self) -> str:
        return f'{self._op1} {self.operator} {self._op2}'

    def _equals(self, other: Comparison) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('operator', self.operator.value)]

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    @staticmethod
    def _sql_compatible(t1: ts.ColumnType, t2: ts.ColumnType) -> bool:
        """True if t1 and t2 can be compared directly in SQL.

        Postgres applies its own implicit-cast rules across the numeric hierarchy and between
        date/timestamp, so we let those through with no explicit CAST. Anything else (e.g.
        string vs. json, image vs. anything) falls back to Python evaluation in FilterNode.
        """
        if str(t1.to_sa_type()) == str(t2.to_sa_type()):
            return True
        if t1.is_numeric_type() and t2.is_numeric_type():
            return True
        return (t1.is_date_type() or t1.is_timestamp_type()) and (t2.is_date_type() or t2.is_timestamp_type())

    def _index_value_col(self) -> sql.Column | None:
        """The value column of a B-tree index that can answer this comparison, or None if no such index is present"""
        import pixeltable.index as index

        if not self.is_search_arg_comparison:
            return None
        assert isinstance(self._op1, ColumnRef)
        col = self._op1.col
        tbl = col.get_tbl()
        if not tbl.supports_idxs:
            return None
        idx_info = tbl.find_btree_index(col)
        if idx_info is None or idx_info.val_col is None:
            return None
        if (
            isinstance(self._op2, Literal)
            and self._op2.col_type.is_string_type()
            and len(self._op2.val) >= index.BtreeIndex.MAX_STRING_LEN
        ):
            # Strings are truncated in the value column, so a value column can be used only for comparisons with
            # literals shorter than the limit.
            return None
        return idx_info.val_col.sa_col

    def sql_expr(self, sql_elements: SqlElementCache) -> sql.ColumnElement | None:
        import pixeltable.index as index

        if not self._sql_compatible(self._op1.col_type, self._op2.col_type):
            # e.g. string vs. json, or image vs. anything
            return None

        right = sql_elements.get(self._op2)
        if right is None:
            return None
        val_col = self._index_value_col()
        if val_col is not None and isinstance(self._op2, Variable) and self._op1.col_type.is_string_type():
            # It's a string comparison with a Variable (whose length is unknown in compile time), and there is a B-tree
            # index on truncated values. Due to truncation, we cannot rely on the index value column alone for
            # comparison, but we can optimize with it.
            if self.operator == ComparisonOperator.NE:
                # A B-tree index can't help with !=
                val_col = None
            else:
                stored_col = sql_elements.get(self._op1)
                if stored_col is None:
                    return None
                truncated = sql.func.left(right, index.BtreeIndex.MAX_STRING_LEN)
                if self.operator == ComparisonOperator.EQ:
                    idx_filter = val_col == truncated
                elif self.operator in (ComparisonOperator.LT, ComparisonOperator.LE):
                    idx_filter = val_col <= truncated
                else:
                    idx_filter = val_col >= truncated
                return sql.and_(idx_filter, self._sql_comparison(stored_col, right))

        left = val_col if val_col is not None else sql_elements.get(self._op1)
        if left is None:
            return None
        return self._sql_comparison(left, right)

    def _sql_comparison(self, left: sql.ColumnElement, right: sql.ColumnElement) -> sql.ColumnElement:
        if self.operator == ComparisonOperator.LT:
            return left < right
        if self.operator == ComparisonOperator.LE:
            return left <= right
        if self.operator == ComparisonOperator.EQ:
            return left == right
        if self.operator == ComparisonOperator.NE:
            return left != right
        if self.operator == ComparisonOperator.GT:
            return left > right
        if self.operator == ComparisonOperator.GE:
            return left >= right

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        left = data_row[self._op1.slot_idx]
        right = data_row[self._op2.slot_idx]

        if self.operator == ComparisonOperator.LT:
            data_row[self.slot_idx] = left < right
        elif self.operator == ComparisonOperator.LE:
            data_row[self.slot_idx] = left <= right
        elif self.operator == ComparisonOperator.EQ:
            data_row[self.slot_idx] = left == right
        elif self.operator == ComparisonOperator.NE:
            data_row[self.slot_idx] = left != right
        elif self.operator == ComparisonOperator.GT:
            data_row[self.slot_idx] = left > right
        elif self.operator == ComparisonOperator.GE:
            data_row[self.slot_idx] = left >= right

    def _as_dict(self) -> dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr], tbl_versions: Any = None) -> Comparison:
        assert 'operator' in d
        return cls(ComparisonOperator(d['operator']), components[0], components[1])

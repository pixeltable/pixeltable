from __future__ import annotations

from typing import Any, Optional

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .globals import ComparisonOperator
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class Comparison(Expr):
    is_search_arg_comparison: bool
    operator: ComparisonOperator

    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        from pixeltable import index

        super().__init__(ts.BoolType())
        self.operator = operator

        # if this is a comparison of a column to a literal (ie, could be used as a search argument in an index lookup),
        # normalize it to <column> <operator> <literal>.
        if isinstance(op1, ColumnRef) and isinstance(op2, Literal):
            self.is_search_arg_comparison = True
            self.components = [op1, op2]
        elif isinstance(op1, Literal) and isinstance(op2, ColumnRef):
            self.is_search_arg_comparison = True
            self.components = [op2, op1]
            self.operator = self.operator.reverse()
        else:
            self.is_search_arg_comparison = False
            self.components = [op1, op2]

        if (
            self.is_search_arg_comparison
            and self._op2.col_type.is_string_type()
            and len(self._op2.val) >= index.BtreeIndex.MAX_STRING_LEN
        ):
            # we can't use an index for this after all
            raise excs.Error(
                f'String literal too long for comparison against indexed column {self._op1.col.name!r} '
                f'(max length is {index.BtreeIndex.MAX_STRING_LEN - 1})'
            )

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

    def sql_expr(self, sql_elements: SqlElementCache) -> Optional[sql.ColumnElement]:
        from pixeltable import index

        if str(self._op1.col_type.to_sa_type()) != str(self._op2.col_type.to_sa_type()):
            # Comparing columns of different SQL types (e.g., string vs. json); this can only be done in Python
            # TODO(aaron-siegel): We may be able to handle some cases in SQL by casting one side to the other's type
            return None

        left = sql_elements.get(self._op1)
        if self.is_search_arg_comparison:
            # reference the index value column if there is an index and this is not a snapshot
            # (indices don't apply to snapshots)
            tbl = self._op1.col.tbl
            idx_info = [
                info for info in self._op1.col.get_idx_info().values() if isinstance(info.idx, index.BtreeIndex)
            ]
            if len(idx_info) > 0 and not tbl.is_snapshot:
                # there shouldn't be multiple B-tree indices on a column
                assert len(idx_info) == 1
                left = idx_info[0].val_col.sa_col

        right = sql_elements.get(self._op2)
        if left is None or right is None:
            return None

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
    def _from_dict(cls, d: dict, components: list[Expr]) -> Comparison:
        assert 'operator' in d
        return cls(ComparisonOperator(d['operator']), components[0], components[1])

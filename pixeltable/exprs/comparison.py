from __future__ import annotations

from typing import Optional, List, Any, Dict, Tuple

import sqlalchemy as sql

from .column_ref import ColumnRef
from .data_row import DataRow
from .expr import Expr
from .globals import ComparisonOperator
from .literal import Literal
from .predicate import Predicate
from .row_builder import RowBuilder
import pixeltable.exceptions as excs


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
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

        import pixeltable.index as index
        if self.is_search_arg_comparison and self._op2.col_type.is_string_type() \
                and len(self._op2.val) >= index.BtreeIndex.MAX_STRING_LEN:
            # we can't use an index for this after all
            raise excs.Error(
                f'String literal too long for comparison against indexed column {self._op1.col.name!r} '
                f'(max length is {index.BtreeIndex.MAX_STRING_LEN - 1})'
            )

        self.id = self._create_id()

    def __str__(self) -> str:
        return f'{self._op1} {self.operator} {self._op2}'

    def _equals(self, other: Comparison) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> List[Tuple[str, Any]]:
        return super()._id_attrs() + [('operator', self.operator.value)]

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def sql_expr(self) -> Optional[sql.ClauseElement]:
        left = self._op1.sql_expr()
        if self.is_search_arg_comparison:
            # reference the index value column if there is an index and this is not a snapshot
            # (indices don't apply to snapshots)
            tbl = self._op1.col.tbl
            idx_info = self._op1.col.get_idx_info()
            if len(idx_info) > 0 and not tbl.is_snapshot:
                # there shouldn't be multiple B-tree indices on a column
                assert len(idx_info) == 1
                left = next(iter(idx_info.values())).val_col.sa_col

        right = self._op2.sql_expr()
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
        if self.operator == ComparisonOperator.LT:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] < data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.LE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] <= data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.EQ:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] == data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.NE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] != data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.GT:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] > data_row[self._op2.slot_idx]
        elif self.operator == ComparisonOperator.GE:
            data_row[self.slot_idx] = data_row[self._op1.slot_idx] >= data_row[self._op2.slot_idx]

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'operator' in d
        return cls(ComparisonOperator(d['operator']), components[0], components[1])


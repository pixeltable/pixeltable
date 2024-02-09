from __future__ import annotations
from typing import Optional, List, Any, Dict, Tuple

import sqlalchemy as sql

from .globals import ArithmeticOperator
from .expr import Expr
from .data_row import DataRow
from .row_builder import RowBuilder
import pixeltable.exceptions as excs
import pixeltable.catalog as catalog
import pixeltable.type_system as ts


class ArithmeticExpr(Expr):
    """
    Allows arithmetic exprs on json paths
    """
    def __init__(self, operator: ArithmeticOperator, op1: Expr, op2: Expr):
        # TODO: determine most specific common supertype
        if op1.col_type.is_json_type() or op2.col_type.is_json_type():
            # we assume it's a float
            super().__init__(ts.FloatType())
        else:
            super().__init__(ts.ColumnType.supertype(op1.col_type, op2.col_type))
        self.operator = operator
        self.components = [op1, op2]

        # do typechecking after initialization in order for __str__() to work
        if not op1.col_type.is_numeric_type() and not op1.col_type.is_json_type():
            raise excs.Error(f'{self}: {operator} requires numeric types, but {op1} has type {op1.col_type}')
        if not op2.col_type.is_numeric_type() and not op2.col_type.is_json_type():
            raise excs.Error(f'{self}: {operator} requires numeric types, but {op2} has type {op2.col_type}')

        self.id = self._create_id()

    def __str__(self) -> str:
        # add parentheses around operands that are ArithmeticExprs to express precedence
        op1_str = f'({self._op1})' if isinstance(self._op1, ArithmeticExpr) else str(self._op1)
        op2_str = f'({self._op2})' if isinstance(self._op2, ArithmeticExpr) else str(self._op2)
        return f'{op1_str} {str(self.operator)} {op2_str}'

    def _equals(self, other: ArithmeticExpr) -> bool:
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
        right = self._op2.sql_expr()
        if left is None or right is None:
            return None
        if self.operator == ArithmeticOperator.ADD:
            return left + right
        if self.operator == ArithmeticOperator.SUB:
            return left - right
        if self.operator == ArithmeticOperator.MUL:
            return left * right
        if self.operator == ArithmeticOperator.DIV:
            return left / right
        if self.operator == ArithmeticOperator.MOD:
            return left % right

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        op1_val = data_row[self._op1.slot_idx]
        op2_val = data_row[self._op2.slot_idx]
        # check types if we couldn't do that prior to execution
        if self._op1.col_type.is_json_type() and not isinstance(op1_val, int) and not isinstance(op1_val, float):
            raise excs.Error(
                f'{self.operator} requires numeric type, but {self._op1} has type {type(op1_val).__name__}')
        if self._op2.col_type.is_json_type() and not isinstance(op2_val, int) and not isinstance(op2_val, float):
            raise excs.Error(
                f'{self.operator} requires numeric type, but {self._op2} has type {type(op2_val).__name__}')

        if self.operator == ArithmeticOperator.ADD:
            data_row[self.slot_idx] = op1_val + op2_val
        elif self.operator == ArithmeticOperator.SUB:
            data_row[self.slot_idx] = op1_val - op2_val
        elif self.operator == ArithmeticOperator.MUL:
            data_row[self.slot_idx] = op1_val * op2_val
        elif self.operator == ArithmeticOperator.DIV:
            data_row[self.slot_idx] = op1_val / op2_val
        elif self.operator == ArithmeticOperator.MOD:
            data_row[self.slot_idx] = op1_val % op2_val

    def _as_dict(self) -> Dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: Dict, components: List[Expr]) -> Expr:
        assert 'operator' in d
        assert len(components) == 2
        return cls(ArithmeticOperator(d['operator']), components[0], components[1])

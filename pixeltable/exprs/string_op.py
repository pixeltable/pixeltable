from __future__ import annotations

from typing import Any

import sqlalchemy as sql

import pixeltable.exceptions as excs
import pixeltable.type_system as ts

from .data_row import DataRow
from .expr import Expr
from .globals import StringOperator
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class StringOp(Expr):
    """
    Allows operations on strings
    """

    operator: StringOperator

    def __init__(self, operator: StringOperator, op1: Expr, op2: Expr):
        super().__init__(ts.StringType(nullable=op1.col_type.nullable))
        self.operator = operator
        self.components = [op1, op2]
        match operator:
            case StringOperator.CONCAT:
                if not op1.col_type.is_string_type() or not op2.col_type.is_string_type():
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'{self}: {operator} on strings requires `String` and `String`, but operands have types `{op1.col_type}` and `{op2.col_type}`',
                    )
            case StringOperator.REPEAT:
                if not (op1.col_type.is_string_type() and op2.col_type.is_int_type()) and not (
                    op1.col_type.is_int_type() and op2.col_type.is_string_type()
                ):
                    raise excs.RequestError(
                        excs.ErrorCode.TYPE_MISMATCH,
                        f'{self}: {operator} on strings requires `String` and `Int`, but operands have types `{op1.col_type}` and `{op2.col_type}`',
                    )

        self.id = self._create_id()

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def __repr__(self) -> str:
        # add parentheses around operands that are StringOpExpr to express precedence
        op1_str = f'({self._op1})' if isinstance(self._op1, StringOp) else str(self._op1)
        op2_str = f'({self._op2})' if isinstance(self._op2, StringOp) else str(self._op2)
        return f'{op1_str} {self.operator} {op2_str}'

    def _equals(self, other: StringOp) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('operator', self.operator.value)]

    def _substitute(self, spec: dict[Expr, Expr]) -> StringOp:
        return StringOp(self.operator, self._op1.substitute(spec), self._op2.substitute(spec))

    def sql_expr(self, sql_elements: SqlElementCache) -> sql.ColumnElement | None:
        left = sql_elements.get(self._op1)
        right = sql_elements.get(self._op2)
        if left is None or right is None:
            return None
        if self.operator == StringOperator.CONCAT:
            return left.concat(right)
        if self.operator == StringOperator.REPEAT:
            if self._op1.col_type.is_string_type():
                return sql.func.repeat(left.cast(sql.String), right.cast(sql.Integer))
            else:
                return sql.func.repeat(right.cast(sql.String), left.cast(sql.Integer))
        return None

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        op1_val = data_row[self._op1.slot_idx]
        op2_val = data_row[self._op2.slot_idx]
        data_row[self.slot_idx] = self.eval_nullable(op1_val, op2_val)

    def eval_nullable(self, op1_val: str | None, op2_val: int | str | None) -> str | None:
        """
        Return the result of evaluating the expression on two nullable int/float operands,
        None is interpreted as SQL NULL
        """
        if op1_val is None or op2_val is None:
            return None
        return self.eval_non_null(op1_val, op2_val)

    def eval_non_null(self, op1_val: str, op2_val: int | str) -> str:
        """
        Return the result of evaluating the expression on two int/float operands
        """
        if self.operator == StringOperator.CONCAT:
            return op1_val + op2_val
        else:
            return op1_val * op2_val

    def _as_dict(self) -> dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> StringOp:
        assert 'operator' in d
        assert len(components) == 2
        return cls(StringOperator(d['operator']), components[0], components[1])

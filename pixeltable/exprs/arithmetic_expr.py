from __future__ import annotations

from typing import Any

import sqlalchemy as sql

from pixeltable import env, exceptions as excs, type_system as ts

from .data_row import DataRow
from .expr import Expr
from .globals import ArithmeticOperator
from .literal import Literal
from .row_builder import RowBuilder
from .sql_element_cache import SqlElementCache


class ArithmeticExpr(Expr):
    """
    Allows arithmetic exprs on json paths
    """

    operator: ArithmeticOperator

    def __init__(self, operator: ArithmeticOperator, op1: Expr, op2: Expr):
        if op1.col_type.is_json_type() or op2.col_type.is_json_type() or operator == ArithmeticOperator.DIV:
            # we assume it's a float
            super().__init__(ts.FloatType(nullable=(op1.col_type.nullable or op2.col_type.nullable)))
        else:
            super().__init__(op1.col_type.supertype(op2.col_type))
        self.operator = operator
        self.components = [op1, op2]

        # do typechecking after initialization in order for __str__() to work
        if not op1.col_type.is_numeric_type() and not op1.col_type.is_json_type():
            raise excs.Error(f'{self}: {operator} requires numeric types, but {op1} has type {op1.col_type}')
        if not op2.col_type.is_numeric_type() and not op2.col_type.is_json_type():
            raise excs.Error(f'{self}: {operator} requires numeric types, but {op2} has type {op2.col_type}')

        self.id = self._create_id()

    @property
    def _op1(self) -> Expr:
        return self.components[0]

    @property
    def _op2(self) -> Expr:
        return self.components[1]

    def __repr__(self) -> str:
        # add parentheses around operands that are ArithmeticExprs to express precedence
        op1_str = f'({self._op1})' if isinstance(self._op1, ArithmeticExpr) else str(self._op1)
        op2_str = f'({self._op2})' if isinstance(self._op2, ArithmeticExpr) else str(self._op2)
        return f'{op1_str} {self.operator} {op2_str}'

    def _equals(self, other: ArithmeticExpr) -> bool:
        return self.operator == other.operator

    def _id_attrs(self) -> list[tuple[str, Any]]:
        return [*super()._id_attrs(), ('operator', self.operator.value)]

    def sql_expr(self, sql_elements: SqlElementCache) -> sql.ColumnElement | None:
        assert self.col_type.is_int_type() or self.col_type.is_float_type() or self.col_type.is_json_type()
        left = sql_elements.get(self._op1)
        right = sql_elements.get(self._op2)
        if left is None or right is None:
            return None
        if self.operator in (ArithmeticOperator.ADD, ArithmeticOperator.SUB, ArithmeticOperator.MUL):
            if env.Env.get().is_using_cockroachdb and self._op1.col_type != self._op2.col_type:
                if self._op1.col_type != self.col_type:
                    left = sql.cast(left, self.col_type.to_sa_type())
                if self._op2.col_type != self.col_type:
                    right = sql.cast(right, self.col_type.to_sa_type())
            if self.operator == ArithmeticOperator.ADD:
                return left + right
            if self.operator == ArithmeticOperator.SUB:
                return left - right
            if self.operator == ArithmeticOperator.MUL:
                return left * right
        if self.operator == ArithmeticOperator.DIV:
            assert self.col_type.is_float_type()
            # Avoid division by zero errors by converting any zero divisor to NULL.
            # TODO: Should we cast the NULLs to NaNs when they are retrieved back into Python?
            # These casts cause the computation to take place in float units, rather than DECIMAL.
            nullif = sql.cast(sql.func.nullif(right, 0), self.col_type.to_sa_type())
            return sql.cast(left, self.col_type.to_sa_type()) / nullif
        if self.operator == ArithmeticOperator.MOD:
            if self.col_type.is_int_type():
                # Avoid division by zero errors by converting any zero divisor to NULL.
                nullif1 = sql.cast(sql.func.nullif(right, 0), self.col_type.to_sa_type())
                return left % nullif1
            if self.col_type.is_float_type():
                # Postgres does not support modulus for floats
                return None
        if self.operator == ArithmeticOperator.FLOORDIV:
            # Postgres has a DIV operator, but it behaves differently from Python's // operator
            # (Postgres rounds toward 0, Python rounds toward negative infinity)
            # We need the behavior to be consistent, so that expressions will evaluate the same way
            # whether or not their operands can be translated to SQL. These SQL clauses should
            # mimic the behavior of Python's // operator.
            # Avoid division by zero errors by converting any zero divisor to NULL.
            nullif = sql.cast(sql.func.nullif(right, 0), self.col_type.to_sa_type())
            return sql.func.floor(sql.cast(left, self.col_type.to_sa_type()) / nullif)
        raise AssertionError()

    def eval(self, data_row: DataRow, row_builder: RowBuilder) -> None:
        op1_val = data_row[self._op1.slot_idx]
        op2_val = data_row[self._op2.slot_idx]

        # if one or both columns is JsonTyped, we need a dynamic check that they are numeric
        if self._op1.col_type.is_json_type() and op1_val is not None and not isinstance(op1_val, (int, float)):
            raise excs.Error(
                f'{self.operator} requires numeric types, but {self._op1} has type {type(op1_val).__name__}'
            )
        if self._op2.col_type.is_json_type() and op2_val is not None and not isinstance(op2_val, (int, float)):
            raise excs.Error(
                f'{self.operator} requires numeric types, but {self._op2} has type {type(op2_val).__name__}'
            )

        data_row[self.slot_idx] = self.eval_nullable(op1_val, op2_val)

    def eval_nullable(self, op1_val: float | None, op2_val: float | None) -> float | None:
        """
        Return the result of evaluating the expression on two nullable int/float operands,
        None is interpreted as SQL NULL
        """
        if op1_val is None or op2_val is None:
            return None
        return self.eval_non_null(op1_val, op2_val)

    def eval_non_null(self, op1_val: float, op2_val: float) -> float:
        """
        Return the result of evaluating the expression on two int/float operands
        """
        if self.operator == ArithmeticOperator.ADD:
            return op1_val + op2_val
        elif self.operator == ArithmeticOperator.SUB:
            return op1_val - op2_val
        elif self.operator == ArithmeticOperator.MUL:
            return op1_val * op2_val
        elif self.operator == ArithmeticOperator.DIV:
            return op1_val / op2_val
        elif self.operator == ArithmeticOperator.MOD:
            return op1_val % op2_val
        elif self.operator == ArithmeticOperator.FLOORDIV:
            return op1_val // op2_val

    def as_literal(self) -> Literal | None:
        op1_lit = self._op1.as_literal()
        if op1_lit is None:
            return None
        op2_lit = self._op2.as_literal()
        if op2_lit is None:
            return None
        op1_val = op1_lit.val
        assert op1_lit.col_type.is_numeric_type() or op1_val is None
        op2_val = op2_lit.val
        assert op2_lit.col_type.is_numeric_type() or op2_val is None

        return Literal(self.eval_nullable(op1_val, op2_val), self.col_type)  # type: ignore[arg-type]

    def _as_dict(self) -> dict:
        return {'operator': self.operator.value, **super()._as_dict()}

    @classmethod
    def _from_dict(cls, d: dict, components: list[Expr]) -> ArithmeticExpr:
        assert 'operator' in d
        assert len(components) == 2
        return cls(ArithmeticOperator(d['operator']), components[0], components[1])

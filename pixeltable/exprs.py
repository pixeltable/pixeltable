import abc
import datetime
import enum
import inspect
import typing
from typing import Union, Optional, List, Callable, Any

import sqlalchemy as sql

from pixeltable import catalog
from pixeltable import type_system as pt_types
from pixeltable import exceptions as exc

# Python types corresponding to our literal types
LiteralPythonTypes = Union[str, int, float, bool, datetime.datetime, datetime.date]


class ComparisonOperator(enum.Enum):
    LT = 1
    LE = 2
    EQ = 3
    NE = 4
    GT = 5
    GE = 6


class LogicalOperator(enum.Enum):
    AND = 1
    OR = 2
    NOT = 3


class Expr(abc.ABC):
    def __init__(self, col_type: pt_types.ColumnType):
        self.col_type = col_type

    @abc.abstractmethod
    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        pass

    def __lt__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.LT, other)

    def __le__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.LE, other)

    def __eq__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.EQ, other)

    def __ne__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.NE, other)

    def __gt__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.GT, other)

    def __ge__(self, other: object) -> 'Comparison':
        return self._make_comparison(ComparisonOperator.GE, other)

    def _make_comparison(self, op: ComparisonOperator, other: object) -> 'Comparison':
        """
        other: Union[Expr, LiteralPythonTypes]
        """
        # TODO: check for compatibility
        if isinstance(other, Expr):
            return Comparison(op, self, other)
        if isinstance(other, typing.get_args(LiteralPythonTypes)):
            return Comparison(op, self, Literal(other))  # type: ignore[arg-type]
        raise TypeError(f'Other must be Expr or literal: {type(other)}')

    @abc.abstractmethod
    def get_eval_info(self) -> Union[sql.sql.expression.ClauseElement, List['ColumnRef']]:
        """
        Return parameters needed to obtain Expr's value for a sql result row.
        - if ClauseElement, we can compute the value via SQL directly and eval() will not be called
        - if List[ColumnRef], eval() will be called and the corresponding cols passed in as args
        """
        pass

    def eval(self, *args: Any) -> Any:
        # TODO: raise something like InternalError?
        assert False


class ColumnRef(Expr):
    def __init__(self, col: catalog.Column):
        super().__init__(col.col_type)
        self.col = col

    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        return self.col.sa_col

    def get_eval_info(self) -> Union[sql.sql.expression.ClauseElement, List['ColumnRef']]:
        return self.to_sql_expr()


class FunctionCall(Expr):
    def __init__(self, fn: Callable, tbl: catalog.Table, col_type: pt_types.ColumnType):
        super().__init__(col_type)
        self.fn = fn
        # make sure that fn's params are valid col names in tbl
        self.col_args: List[catalog.Column] = []
        for param_name in inspect.signature(fn).parameters:
            if param_name not in tbl.cols_by_name:
                raise exc.OperationalError(
                    f'Lambda needs to reference valid columns in table {tbl.name}: column {param_name} unknown')
            self.col_args.append(tbl.cols_by_name[param_name])

    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        assert False

    def get_eval_info(self) -> Union[sql.sql.expression.ClauseElement, List['ColumnRef']]:
        return [ColumnRef(col) for col in self.col_args]

    def eval(self, *args: Any) -> Any:
        return self.fn(*args)


class Literal(Expr):
    def __init__(self, val: LiteralPythonTypes):
        if isinstance(val, str):
            super().__init__(pt_types.ColumnType.STRING)
        if isinstance(val, int):
            super().__init__(pt_types.ColumnType.INT)
        if isinstance(val, float):
            super().__init__(pt_types.ColumnType.FLOAT)
        if isinstance(val, bool):
            super().__init__(pt_types.ColumnType.BOOL)
        if isinstance(val, datetime.datetime) or isinstance(val, datetime.date):
            super().__init__(pt_types.ColumnType.TIMESTAMP)
        self.val = val

    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        return sql.sql.expression.literal(self.val)

    def get_eval_info(self) -> Union[sql.sql.expression.ClauseElement, List['ColumnRef']]:
        return self.to_sql_expr()


class Predicate(Expr):
    def __init__(self) -> None:
        super().__init__(pt_types.ColumnType.BOOL)

    def __and__(self, other: object) -> 'CompoundPredicate':
        if not isinstance(other, Predicate):
            raise TypeError(f'Other needs to be a predicate: {type(other)}')
        assert isinstance(other, Predicate)
        return CompoundPredicate(LogicalOperator.AND, self, other)

    def __or__(self, other: object) -> 'CompoundPredicate':
        if not isinstance(other, Predicate):
            raise TypeError(f'Other needs to be a predicate: {type(other)}')
        assert isinstance(other, Predicate)
        return CompoundPredicate(LogicalOperator.OR, self, other)

    def __invert__(self) -> 'CompoundPredicate':
        return CompoundPredicate(LogicalOperator.NOT, self)

    def get_eval_info(self) -> Union[sql.sql.expression.ClauseElement, List['ColumnRef']]:
        return self.to_sql_expr()


class CompoundPredicate(Predicate):
    def __init__(self, operator: LogicalOperator, op1: Predicate, op2: Optional[Predicate] = None):
        super().__init__()
        self.operator = operator
        self.op1 = op1
        self.op2 = op2

    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        if self.operator == LogicalOperator.AND:
            assert self.op2 is not None
            return sql.and_(self.op1.to_sql_expr(), self.op2.to_sql_expr())
        if self.operator == LogicalOperator.OR:
            assert self.op2 is not None
            return sql.or_(self.op1.to_sql_expr(), self.op2.to_sql_expr())
        if self.operator == LogicalOperator.NOT:
            assert self.op2 is None
            return sql.not_(self.op1.to_sql_expr())


class Comparison(Predicate):
    def __init__(self, operator: ComparisonOperator, op1: Expr, op2: Expr):
        super().__init__()
        self.operator = operator
        self.op1 = op1
        self.op2 = op2

    def to_sql_expr(self) -> sql.sql.expression.ClauseElement:
        if self.operator == ComparisonOperator.LT:
            return self.op1.to_sql_expr() < self.op2.to_sql_expr()
        if self.operator == ComparisonOperator.LE:
            return self.op1.to_sql_expr() <= self.op2.to_sql_expr()
        if self.operator == ComparisonOperator.EQ:
            return self.op1.to_sql_expr() == self.op2.to_sql_expr()
        if self.operator == ComparisonOperator.NE:
            return self.op1.to_sql_expr() != self.op2.to_sql_expr()
        if self.operator == ComparisonOperator.GT:
            return self.op1.to_sql_expr() > self.op2.to_sql_expr()
        if self.operator == ComparisonOperator.GE:
            return self.op1.to_sql_expr() >= self.op2.to_sql_expr()

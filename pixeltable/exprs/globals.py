from __future__ import annotations

import datetime
import enum
from typing import Union

# Python types corresponding to our literal types
LiteralPythonTypes = Union[str, int, float, bool, datetime.datetime, datetime.date]


def print_slice(s: slice) -> str:
    start_str = f'{str(s.start) if s.start is not None else ""}'
    stop_str = f'{str(s.stop) if s.stop is not None else ""}'
    step_str = f'{str(s.step) if s.step is not None else ""}'
    return f'{start_str}:{stop_str}{":" if s.step is not None else ""}{step_str}'


class ComparisonOperator(enum.Enum):
    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5

    def __str__(self) -> str:
        if self == self.LT:
            return '<'
        if self == self.LE:
            return '<='
        if self == self.EQ:
            return '=='
        if self == self.NE:
            return '!='
        if self == self.GT:
            return '>'
        if self == self.GE:
            return '>='
        raise AssertionError()

    def reverse(self) -> ComparisonOperator:
        if self == self.LT:
            return self.GT
        if self == self.LE:
            return self.GE
        if self == self.GT:
            return self.LT
        if self == self.GE:
            return self.LE
        return self


class LogicalOperator(enum.Enum):
    AND = 0
    OR = 1
    NOT = 2

    def __str__(self) -> str:
        if self == self.AND:
            return '&'
        if self == self.OR:
            return '|'
        if self == self.NOT:
            return '~'
        raise AssertionError()


class ArithmeticOperator(enum.Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    DIV = 3
    MOD = 4
    FLOORDIV = 5

    def __str__(self) -> str:
        if self == self.ADD:
            return '+'
        if self == self.SUB:
            return '-'
        if self == self.MUL:
            return '*'
        if self == self.DIV:
            return '/'
        if self == self.MOD:
            return '%'
        if self == self.FLOORDIV:
            return '//'
        raise AssertionError()


class StringOperator(enum.Enum):
    CONCAT = 0
    REPEAT = 1

    def __str__(self) -> str:
        if self == self.CONCAT:
            return '+'
        if self == self.REPEAT:
            return '*'
        raise AssertionError()

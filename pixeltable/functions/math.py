"""
Pixeltable UDFs for mathematical operations.

Example:

>>> import pixeltable as pxt
>>> t = pxt.get_table(...)
>>> t.select(t.float_col.floor()).collect()
"""

import builtins
import math

import sqlalchemy as sql

import pixeltable as pxt


@pxt.udf(is_method=True)
def abs(self: float) -> float:
    """
    Return the absolute value of the given number.

    Equivalent to Python [`builtins.abs()`](https://docs.python.org/3/library/functions.html#abs).
    """
    return builtins.abs(self)


@abs.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.abs(self)


@pxt.udf(is_method=True)
def ceil(self: float) -> float:
    """
    Return the ceiling of the given number.

    Equivalent to Python [`float(math.ceil(self))`](https://docs.python.org/3/library/math.html#math.ceil) if `self`
    is finite, or `self` itself if `self` is infinite. (This is slightly different from the default behavior of
    `math.ceil(self)`, which always returns an `int` and raises an error if `self` is infinite. The behavior in
    Pixeltable generalizes the Python operator and is chosen to align with the SQL standard.)
    """
    # This ensures the same behavior as SQL
    if math.isfinite(self):
        return float(math.ceil(self))
    else:
        return self


@ceil.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.ceiling(self, type_=sql.Float)


@pxt.udf(is_method=True)
def floor(self: float) -> float:
    """
    Return the floor of the given number.

    Equivalent to Python [`float(math.floor(self))`](https://docs.python.org/3/library/math.html#math.floor) if `self`
    is finite, or `self` itself if `self` is infinite. (This is slightly different from the default behavior of
    `math.floor(self)`, which always returns an `int` and raises an error if `self` is infinite. The behavior of
    Pixeltable generalizes the Python operator and is chosen to align with the SQL standard.)
    """
    # This ensures the same behavior as SQL
    if math.isfinite(self):
        return float(math.floor(self))
    else:
        return self


@floor.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.floor(self, type_=sql.Float)


@pxt.udf(is_method=True)
def round(self: float, digits: int | None = None) -> float:
    """
    Round a number to a given precision in decimal digits.

    Equivalent to Python [`builtins.round(self, digits or 0)`](https://docs.python.org/3/library/functions.html#round).
    Note that if `digits` is not specified, the behavior matches `builtins.round(self, 0)` rather than
    `builtins.round(self)`; this ensures that the return type is always `float` (as in SQL) rather than `int`.
    """
    # Set digits explicitly to 0 to guarantee a return type of float; this ensures the same behavior as SQL
    return builtins.round(self, digits or 0)


@round.to_sql
def _(self: sql.ColumnElement, digits: sql.ColumnElement | None = None) -> sql.ColumnElement:
    if digits is None:
        # without type_, SQLAlchemy leaves the element's type as NullType
        return sql.func.round(self, type_=sql.Float)
    else:
        return sql.cast(sql.func.round(sql.cast(self, sql.Numeric), sql.cast(digits, sql.Integer)), sql.Float)


@pxt.udf(is_method=True)
def pow(self: float, other: float) -> float:
    """
    Raise `self` to the power of `other`.

    Equivalent to Python [`builtins.pow()`](https://docs.python.org/3/library/functions.html#pow).
    """
    return self**other


@pow.to_sql
def _(self: sql.ColumnElement, other: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.pow(self, other)


@pxt.udf(is_method=True)
def sqrt(self: float) -> float:
    """
    Return the square root of the given number.

    Equivalent to Python [`math.sqrt()`](https://docs.python.org/3/library/math.html#math.sqrt).
    """
    return math.sqrt(self)


@sqrt.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.sqrt(self)


@pxt.udf(is_method=True)
def exp(self: float) -> float:
    """
    Return `e` raised to the power of the given number.

    Equivalent to Python [`math.exp()`](https://docs.python.org/3/library/math.html#math.exp).
    """
    return math.exp(self)


@exp.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.exp(self)


@pxt.udf(is_method=True)
def log(self: float) -> float:
    """
    Return the natural logarithm of the given number.

    Equivalent to Python [`math.log()`](https://docs.python.org/3/library/math.html#math.log).
    """
    return math.log(self)


@log.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.ln(self)


@pxt.udf(is_method=True)
def log10(self: float) -> float:
    """
    Return the base-10 logarithm of the given number.

    Equivalent to Python [`math.log10()`](https://docs.python.org/3/library/math.html#math.log10).
    """
    return math.log10(self)


@log10.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.log10(self)


@pxt.udf(is_method=True)
def bitwise_and(self: int, other: int) -> int:
    """
    Bitwise AND of two integers.

    Equivalent to Python
    [`self & other`](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types).
    """
    return self & other


@bitwise_and.to_sql
def _(self: sql.ColumnElement, other: sql.ColumnElement) -> sql.ColumnElement:
    return self.bitwise_and(other)


@pxt.udf(is_method=True)
def bitwise_or(self: int, other: int) -> int:
    """
    Bitwise OR of two integers.

    Equivalent to Python
    [`self | other`](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types).
    """
    return self | other


@bitwise_or.to_sql
def _(self: sql.ColumnElement, other: sql.ColumnElement) -> sql.ColumnElement:
    return self.bitwise_or(other)


@pxt.udf(is_method=True)
def bitwise_xor(self: int, other: int) -> int:
    """
    Bitwise XOR of two integers.

    Equivalent to Python
    [`self ^ other`](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types).
    """
    return self ^ other


@bitwise_xor.to_sql
def _(self: sql.ColumnElement, other: sql.ColumnElement) -> sql.ColumnElement:
    return self.bitwise_xor(other)


@pxt.udf(is_method=True)
def to_int(self: float) -> int:
    """
    Convert a float to an integer by truncating toward zero.

    Equivalent to Python [`int(self)`](https://docs.python.org/3/library/functions.html#int). Raises an error for
    infinity, NaN, and values outside the range of `Int`.

    To round down, up or to the nearest integer instead, apply [`floor()`][pixeltable.functions.math.floor],
    [`ceil()`][pixeltable.functions.math.ceil] or [`round()`][pixeltable.functions.math.round] first, as in
    `t.x.floor().to_int()`.
    """
    result = builtins.int(self)
    if not -(2**63) <= result < 2**63:
        raise OverflowError(f'{self} is outside the range of Int')
    return result


@to_int.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement | None:
    if isinstance(self.type, sql.Integer):
        return sql.cast(self, sql.BigInteger)
    if isinstance(self.type, sql.Float):
        return sql.cast(sql.func.trunc(self), sql.BigInteger)
    # trunc() rejects a BOOLEAN and converts a BIGINT to DOUBLE PRECISION, which is inexact above 2**53
    return None


@pxt.udf(is_method=True)
def to_float(self: int) -> float:
    """
    Convert an integer to a float.

    Equivalent to Python [`float(self)`](https://docs.python.org/3/library/functions.html#float).
    """
    return builtins.float(self)


@to_float.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.cast(self, sql.Float)


__all__ = [
    'abs',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'ceil',
    'exp',
    'floor',
    'log',
    'log10',
    'pow',
    'round',
    'sqrt',
    'to_float',
    'to_int',
]


def __dir__() -> list[str]:
    return __all__

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
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.abs(self, type_=self.type)


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
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.pow(self, other, type_=sql.Float)


@pxt.udf(is_method=True)
def sqrt(self: float) -> float:
    """
    Return the square root of the given number.

    Equivalent to Python [`math.sqrt()`](https://docs.python.org/3/library/math.html#math.sqrt).
    """
    return math.sqrt(self)


@sqrt.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.sqrt(self, type_=sql.Float)


@pxt.udf(is_method=True)
def exp(self: float) -> float:
    """
    Return `e` raised to the power of the given number.

    Equivalent to Python [`math.exp()`](https://docs.python.org/3/library/math.html#math.exp).
    """
    return math.exp(self)


@exp.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.exp(self, type_=sql.Float)


@pxt.udf(is_method=True)
def log(self: float) -> float:
    """
    Return the natural logarithm of the given number.

    Equivalent to Python [`math.log()`](https://docs.python.org/3/library/math.html#math.log).
    """
    return math.log(self)


@log.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.ln(self, type_=sql.Float)


@pxt.udf(is_method=True)
def log10(self: float) -> float:
    """
    Return the base-10 logarithm of the given number.

    Equivalent to Python [`math.log10()`](https://docs.python.org/3/library/math.html#math.log10).
    """
    return math.log10(self)


@log10.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.log10(self, type_=sql.Float)


@pxt.udf(is_method=True)
def sin(self: float) -> float:
    """
    Return the sine of the given angle, in radians.

    Equivalent to Python [`math.sin()`](https://docs.python.org/3/library/math.html#math.sin).

    Raises an error for infinity.
    """
    return math.sin(self)


@sin.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.sin(self, type_=sql.Float)


@pxt.udf(is_method=True)
def cos(self: float) -> float:
    """
    Return the cosine of the given angle, in radians.

    Equivalent to Python [`math.cos()`](https://docs.python.org/3/library/math.html#math.cos).

    Raises an error for infinity.
    """
    return math.cos(self)


@cos.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.cos(self, type_=sql.Float)


@pxt.udf(is_method=True)
def tan(self: float) -> float:
    """
    Return the tangent of the given angle, in radians.

    Equivalent to Python [`math.tan()`](https://docs.python.org/3/library/math.html#math.tan).

    Raises an error for infinity.
    """
    return math.tan(self)


@tan.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.tan(self, type_=sql.Float)


@pxt.udf(is_method=True)
def asin(self: float) -> float:
    """
    Return the arc sine of the given number, in radians.

    Equivalent to Python [`math.asin()`](https://docs.python.org/3/library/math.html#math.asin).

    Raises an error if the input is outside the range `[-1, 1]`.
    """
    return math.asin(self)


@asin.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.asin(self, type_=sql.Float)


@pxt.udf(is_method=True)
def acos(self: float) -> float:
    """
    Return the arc cosine of the given number, in radians.

    Equivalent to Python [`math.acos()`](https://docs.python.org/3/library/math.html#math.acos).

    Raises an error if the input is outside the range `[-1, 1]`.
    """
    return math.acos(self)


@acos.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.acos(self, type_=sql.Float)


@pxt.udf(is_method=True)
def atan(self: float) -> float:
    """
    Return the arc tangent of the given number, in radians.

    Equivalent to Python [`math.atan()`](https://docs.python.org/3/library/math.html#math.atan).
    """
    return math.atan(self)


@atan.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.atan(self, type_=sql.Float)


@pxt.udf(is_method=True)
def degrees(self: float) -> float:
    """
    Convert an angle from radians to degrees.

    Equivalent to Python [`math.degrees()`](https://docs.python.org/3/library/math.html#math.degrees).
    """
    return math.degrees(self)


@degrees.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.degrees(self, type_=sql.Float)


@pxt.udf(is_method=True)
def radians(self: float) -> float:
    """
    Convert an angle from degrees to radians.

    Equivalent to Python [`math.radians()`](https://docs.python.org/3/library/math.html#math.radians).
    """
    return math.radians(self)


@radians.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # without type_, SQLAlchemy leaves the element's type as NullType
    return sql.func.radians(self, type_=sql.Float)


@pxt.udf(is_method=True)
def sign(self: float) -> float:
    """
    Return the sign of the given number: `-1.0` if it is negative, `1.0` if it is positive, and `0.0` if it is zero.

    NaN is returned unchanged.
    """
    if math.isnan(self):
        return self
    return builtins.float((self > 0) - (self < 0))


@sign.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    # Postgres' sign() returns 0 for NaN; in Postgres, unlike in IEEE 754, NaN compares equal to NaN
    return sql.case(
        (self == sql.literal_column("'NaN'::double precision"), sql.cast(self, sql.Float)),
        else_=sql.func.sign(self, type_=sql.Float),
    )


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
    # sql.Float is a subclass of sql.Numeric in SQLAlchemy 2.0, but not in 2.1
    if isinstance(self.type, (sql.Numeric, sql.Float)):
        return sql.cast(sql.func.trunc(self), sql.BigInteger)
    if isinstance(self.type, sql.Boolean):
        # Postgres casts a boolean only to integer
        return sql.cast(sql.cast(self, sql.Integer), sql.BigInteger)
    # trunc() converts an untyped BIGINT to DOUBLE PRECISION, which is inexact above 2**53
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
    'acos',
    'asin',
    'atan',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'ceil',
    'cos',
    'degrees',
    'exp',
    'floor',
    'log',
    'log10',
    'pow',
    'radians',
    'round',
    'sign',
    'sin',
    'sqrt',
    'tan',
    'to_float',
    'to_int',
]


def __dir__() -> list[str]:
    return __all__

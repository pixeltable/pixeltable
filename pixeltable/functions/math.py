"""
Pixeltable [UDFs](https://pixeltable.readme.io/docs/user-defined-functions-udfs) for mathematical operations.

Example:
```python
import pixeltable as pxt

t = pxt.get_table(...)
t.select(t.float_col.floor()).collect()
```
"""

import builtins
import math
from typing import Optional

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


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
    return sql.func.ceiling(self)


@pxt.udf(is_method=True)
def floor(self: float) -> float:
    """
    Return the ceiling of the given number.

    Equivalent to Python [`float(math.floor(self))`](https://docs.python.org/3/library/math.html#math.ceil) if `self`
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
    return sql.func.floor(self)


@pxt.udf(is_method=True)
def round(self: float, digits: Optional[int] = None) -> float:
    """
    Round a number to a given precision in decimal digits.

    Equivalent to Python [`builtins.round(self, digits or 0)`](https://docs.python.org/3/library/functions.html#round).
    Note that if `digits` is not specified, the behavior matches `builtins.round(self, 0)` rather than
    `builtins.round(self)`; this ensures that the return type is always `float` (as in SQL) rather than `int`.
    """
    # Set digits explicitly to 0 to guarantee a return type of float; this ensures the same behavior as SQL
    return builtins.round(self, digits or 0)


@round.to_sql
def _(self: sql.ColumnElement, digits: Optional[sql.ColumnElement] = None) -> sql.ColumnElement:
    if digits is None:
        return sql.func.round(self)
    else:
        return sql.func.round(sql.cast(self, sql.Numeric), sql.cast(digits, sql.Integer))


@pxt.udf(is_method=True)
def pow(self: int, other: int) -> float:
    """
    Raise `self` to the power of `other`.

    Equivalent to Python [`self ** other`](https://docs.python.org/3/library/functions.html#pow).
    """
    return self**other


@pow.to_sql
def _(self: sql.ColumnElement, other: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.pow(self, other)


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


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

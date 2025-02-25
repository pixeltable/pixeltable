# noqa: A005

import builtins
import math
from typing import Optional

import sqlalchemy as sql

import pixeltable as pxt
from pixeltable.utils.code import local_public_names


@pxt.udf(is_method=True)
def abs(self: float) -> float:  # noqa: A001
    return builtins.abs(self)


@abs.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.abs(self)


@pxt.udf(is_method=True)
def ceil(self: float) -> float:
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
    # This ensures the same behavior as SQL
    if math.isfinite(self):
        return float(math.floor(self))
    else:
        return self


@floor.to_sql
def _(self: sql.ColumnElement) -> sql.ColumnElement:
    return sql.func.floor(self)


@pxt.udf(is_method=True)
def round(self: float, digits: Optional[int] = None) -> float:  # noqa: A001
    # Set digits explicitly to 0 to guarantee a return type of float; this ensures the same behavior as SQL
    return builtins.round(self, digits or 0)


@round.to_sql
def _(self: sql.ColumnElement, digits: Optional[sql.ColumnElement] = None) -> sql.ColumnElement:
    if digits is None:
        return sql.func.round(self)
    else:
        return sql.func.round(sql.cast(self, sql.Numeric), sql.cast(digits, sql.Integer))


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

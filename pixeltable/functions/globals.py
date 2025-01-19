import builtins
from typing import _GenericAlias  # type: ignore[attr-defined]
from typing import Optional, Union
import typing
from io import StringIO

import sqlalchemy as sql

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils.code import local_public_names


# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: Union[ts.ColumnType, type, _GenericAlias]) -> exprs.Expr:
    expr.col_type = ts.ColumnType.normalize_type(target_type)
    return expr


T = typing.TypeVar('T')


@func.uda(allows_window=True, type_substitutions=({T: Optional[int]}, {T: Optional[float]}))  # type: ignore[misc]
class sum(func.Aggregator, typing.Generic[T]):
    """Sums the selected integers or floats."""
    def __init__(self):
        self.sum: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val  # type: ignore[operator]

    def value(self) -> T:
        return self.sum


@sum.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    # This can produce a Decimal. We are deliberately avoiding an explicit cast to a Bigint here, because that can
    # cause overflows in Postgres. We're instead doing the conversion to the target type in SqlNode.__iter__().
    return sql.sql.func.sum(val)


@func.uda(
    allows_window=True,
    # Allow counting non-null values of any type
    # TODO: I couldn't include "Array" because we don't have a way to represent a generic array (of arbitrary dimension).
    # TODO: should we have an "Any" type that can be used here?
    type_substitutions=tuple(
        {T: Optional[t]}  # type: ignore[misc]
        for t in (ts.String, ts.Int, ts.Float, ts.Bool, ts.Timestamp,
                  ts.Json, ts.Image, ts.Video, ts.Audio, ts.Document)
    ),
)
class count(func.Aggregator, typing.Generic[T]):
    def __init__(self):
        self.count = 0

    def update(self, val: T) -> None:
        if val is not None:
            self.count += 1

    def value(self) -> int:
        return self.count


@count.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.count(val)


@func.uda(
    allows_window=True,
    type_substitutions=tuple({T: Optional[t]} for t in (str, int, float, bool, ts.Timestamp))  # type: ignore[misc]
)
class min(func.Aggregator, typing.Generic[T]):
    def __init__(self):
        self.val: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.min(self.val, val)  # type: ignore[call-overload]

    def value(self) -> T:
        return self.val


@min.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    if val.type.python_type == bool:
        # TODO: min/max aggregation of booleans is not supported in Postgres (but it is in Python).
        # Right now we simply force the computation to be done in Python; we might consider implementing an alternate
        # way of doing it in SQL. (min/max of booleans is simply logical and/or, respectively.)
        return None
    return sql.sql.func.min(val)


@func.uda(
    allows_window=True,
    type_substitutions=tuple({T: Optional[t]} for t in (str, int, float, bool, ts.Timestamp))  # type: ignore[misc]
)
class max(func.Aggregator, typing.Generic[T]):
    def __init__(self):
        self.val: T = None

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.max(self.val, val)  # type: ignore[call-overload]

    def value(self) -> T:
        return self.val


@max.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    if val.type.python_type == bool:
        # TODO: see comment in @min.to_sql.
        return None
    return sql.sql.func.max(val)


@func.uda(type_substitutions=({T: Optional[int]}, {T: Optional[float]}))  # type: ignore[misc]
class mean(func.Aggregator, typing.Generic[T]):
    def __init__(self):
        self.sum: T = None
        self.count = 0

    def update(self, val: T) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val  # type: ignore[operator]
        self.count += 1

    def value(self) -> Optional[float]:  # Always a float
        if self.count == 0:
            return None
        return self.sum / self.count  # type: ignore[operator]


@mean.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.avg(val)


@func.uda  # type: ignore[misc]
class str_join(func.Aggregator):
    sep: str
    combined: StringIO
    needs_sep: bool

    def __init__(self, sep: str = ' '):
        self.sep = sep
        self.combined = StringIO()
        self.needs_sep = False

    def update(self, val: str) -> None:
        if self.needs_sep:
            self.combined.write(self.sep)
        else:
            self.needs_sep = True
        self.combined.write(val)

    def value(self) -> str:  # Always a float
        result = self.combined.getvalue()
        self.combined.close()
        return result


@str_join.to_sql
def _(val: sql.ColumnElement, sep: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.aggregate_strings(val, sep)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

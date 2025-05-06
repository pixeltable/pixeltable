import builtins
import typing
from typing import Any, Callable, Optional, Union

import sqlalchemy as sql

from pixeltable import exceptions as excs, exprs, func, type_system as ts
from pixeltable.utils.code import local_public_names

from typing import _GenericAlias  # type: ignore[attr-defined]  # isort: skip


# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: Union[ts.ColumnType, type, _GenericAlias]) -> exprs.Expr:
    expr.col_type = ts.ColumnType.normalize_type(target_type)
    return expr


T = typing.TypeVar('T')


@func.uda(allows_window=True, type_substitutions=({T: Optional[int]}, {T: Optional[float]}))  # type: ignore[misc]
class sum(func.Aggregator, typing.Generic[T]):
    """Sums the selected integers or floats."""

    def __init__(self) -> None:
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
    # TODO: should we have an "Any" type that can be used here?
    type_substitutions=tuple({T: Optional[t]} for t in ts.ALL_PIXELTABLE_TYPES),  # type: ignore[misc]
)
class count(func.Aggregator, typing.Generic[T]):
    def __init__(self) -> None:
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
    type_substitutions=tuple({T: Optional[t]} for t in (str, int, float, bool, ts.Timestamp)),  # type: ignore[misc]
)
class min(func.Aggregator, typing.Generic[T]):
    def __init__(self) -> None:
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
    if val.type.python_type is bool:
        # TODO: min/max aggregation of booleans is not supported in Postgres (but it is in Python).
        # Right now we simply force the computation to be done in Python; we might consider implementing an alternate
        # way of doing it in SQL. (min/max of booleans is simply logical and/or, respectively.)
        return None
    return sql.sql.func.min(val)


@func.uda(
    allows_window=True,
    type_substitutions=tuple({T: Optional[t]} for t in (str, int, float, bool, ts.Timestamp)),  # type: ignore[misc]
)
class max(func.Aggregator, typing.Generic[T]):
    def __init__(self) -> None:
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
    if val.type.python_type is bool:
        # TODO: see comment in @min.to_sql.
        return None
    return sql.sql.func.max(val)


@func.uda(type_substitutions=({T: Optional[int]}, {T: Optional[float]}))  # type: ignore[misc]
class mean(func.Aggregator, typing.Generic[T]):
    def __init__(self) -> None:
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


def map(expr: exprs.Expr, fn: Callable[[exprs.Expr], Any]) -> exprs.Expr:
    target_expr: exprs.Expr
    try:
        target_expr = exprs.Expr.from_object(fn(exprs.json_path.RELATIVE_PATH_ROOT))
    except Exception as e:
        raise excs.Error(
            'Failed to evaluate map function. '
            '(The `fn` argument to `map()` must produce a valid Pixeltable expression.)'
        ) from e
    return exprs.JsonMapper(expr, target_expr)


__all__ = local_public_names(__name__)


def __dir__() -> list[str]:
    return __all__

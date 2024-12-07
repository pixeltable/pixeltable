import builtins
from typing import _GenericAlias  # type: ignore[attr-defined]
from typing import Optional, Union

import sqlalchemy as sql

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils.code import local_public_names


# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: Union[ts.ColumnType, type, _GenericAlias]) -> exprs.Expr:
    expr.col_type = ts.ColumnType.normalize_type(target_type)
    return expr


@func.uda(allows_window=True)
class sum(func.Aggregator):
    """Sums the selected integers or floats."""
    def __init__(self):
        self.sum: Optional[int] = None

    def update(self, val: Optional[int]) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val

    def value(self) -> Optional[int]:
        return self.sum


@sum.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    # This can produce a Decimal. We are deliberately avoiding an explicit cast to a Bigint here, because that can
    # cause overflows in Postgres. We're instead doing the conversion to the target type in SqlNode.__iter__().
    return sql.sql.func.sum(val)


@func.uda(allows_window=True)
class count(func.Aggregator):
    def __init__(self):
        self.count = 0

    def update(self, val: Optional[int]) -> None:
        if val is not None:
            self.count += 1

    def value(self) -> int:
        return self.count


@count.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.count(val)


@func.uda(allows_window=True)
class min(func.Aggregator):
    def __init__(self):
        self.val: Optional[int] = None

    def update(self, val: Optional[int]) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.min(self.val, val)

    def value(self) -> Optional[int]:
        return self.val


@min.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.min(val)


@func.uda(allows_window=True)
class max(func.Aggregator):
    def __init__(self):
        self.val: Optional[int] = None

    def update(self, val: Optional[int]) -> None:
        if val is None:
            return
        if self.val is None:
            self.val = val
        else:
            self.val = builtins.max(self.val, val)

    def value(self) -> Optional[int]:
        return self.val


@max.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.max(val)


@func.uda
class mean(func.Aggregator):
    def __init__(self):
        self.sum: Optional[int] = None
        self.count = 0

    def update(self, val: Optional[int]) -> None:
        if val is None:
            return
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val
        self.count += 1

    def value(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count


@mean.to_sql
def _(val: sql.ColumnElement) -> Optional[sql.ColumnElement]:
    return sql.sql.func.avg(val)


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

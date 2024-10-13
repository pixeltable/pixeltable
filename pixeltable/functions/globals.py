from typing import Optional, Union, _GenericAlias

import pixeltable.func as func
import pixeltable.type_system as ts
from pixeltable import exprs
from pixeltable.utils.code import local_public_names


# TODO: remove and replace calls with astype()
def cast(expr: exprs.Expr, target_type: Union[ts.ColumnType, type, _GenericAlias]) -> exprs.Expr:
    expr.col_type = ts.ColumnType.normalize_type(target_type)
    return expr


@func.uda(update_types=[ts.IntType()], value_type=ts.IntType(), allows_window=True, requires_order_by=False)
class sum(func.Aggregator):
    """Sums the selected integers or floats."""
    def __init__(self):
        self.sum: Union[int, float] = 0

    def update(self, val: Union[int, float]) -> None:
        if val is not None:
            self.sum += val

    def value(self) -> Union[int, float]:
        return self.sum

# @sum.to_sql
# def _(val: 'sqlalchemy.ColumnElements') -> Optional['sqlalchemy.ColumnElements']:
#     import sqlalchemy as sql
#     return sql.sql.functions.sum(val)


@func.uda(update_types=[ts.IntType()], value_type=ts.IntType(), allows_window=True, requires_order_by=False)
class count(func.Aggregator):
    def __init__(self):
        self.count = 0

    def update(self, val: int) -> None:
        if val is not None:
            self.count += 1

    def value(self) -> int:
        return self.count


@func.uda(update_types=[ts.FloatType()], value_type=ts.FloatType(nullable=True), allows_window=True, requires_order_by=False)
class max(func.Aggregator):
    def __init__(self):
        self.val = None

    def update(self, val: Optional[float]) -> None:
        if val is not None:
            if self.val is None:
                self.val = val
            else:
                import builtins
                self.val = builtins.max(self.val, val)

    def value(self) -> Optional[float]:
        return self.val


@func.uda(update_types=[ts.FloatType()], value_type=ts.FloatType(nullable=True), allows_window=True, requires_order_by=False)
class min(func.Aggregator):
    def __init__(self):
        self.val = None

    def update(self, val: Optional[float]) -> None:
        if val is not None:
            if self.val is None:
                self.val = val
            else:
                import builtins
                self.val = builtins.min(self.val, val)

    def value(self) -> Optional[float]:
        return self.val


@func.uda(update_types=[ts.IntType()], value_type=ts.FloatType(), allows_window=False, requires_order_by=False)
class mean(func.Aggregator):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val: int) -> None:
        if val is not None:
            self.sum += val
            self.count += 1

    def value(self) -> float:
        if self.count == 0:
            return None
        return self.sum / self.count


__all__ = local_public_names(__name__)


def __dir__():
    return __all__

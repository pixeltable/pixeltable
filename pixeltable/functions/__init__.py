from typing import Callable, List, Optional, Union
import inspect

from pixeltable.type_system import StringType, IntType, JsonType, ColumnType, FloatType
from pixeltable.function import Function
from pixeltable import catalog
from pixeltable import exprs
import pixeltable.exceptions as exc


def udf_call(eval_fn: Callable, return_type: ColumnType, tbl: Optional[catalog.Table]) -> exprs.FunctionCall:
    """
    Interprets eval_fn's parameters to be references to columns in 'tbl' and construct ColumnRefs as args.
    """
    params = inspect.signature(eval_fn).parameters
    if len(params) > 0 and tbl is None:
        raise exc.OperationalError(f'udf_call() is missing tbl parameter')
    args: List[exprs.ColumnRef] = []
    for param_name in params:
        if param_name not in tbl.cols_by_name:
            raise exc.OperationalError(
                (f'udf_call(): lambda argument names need to be valid column names in table {tbl.name}: '
                 f'column {param_name} unknown'))
        args.append(exprs.ColumnRef(tbl.cols_by_name[param_name]))
    fn = Function(return_type, [arg.col_type for arg in args], eval_fn=eval_fn)
    return exprs.FunctionCall(fn, args)

def cast(expr: exprs.Expr, target_type: ColumnType) -> exprs.Expr:
    expr.col_type = target_type
    return expr

dict_map = Function(IntType(), [StringType(), JsonType()], eval_fn=lambda s, d: d[s])


class SumAggregator:
    def __init__(self):
        self.sum: Union[int, float] = 0
    @classmethod
    def make_aggregator(cls) -> 'SumAggregator':
        return cls()
    def update(self, val: Union[int, float]) -> None:
        if val is not None:
            self.sum += val
    def value(self) -> Union[int, float]:
        return self.sum

sum = Function(
    IntType(), [IntType()],
    init_fn=SumAggregator.make_aggregator, update_fn=SumAggregator.update, value_fn=SumAggregator.value)


class CountAggregator:
    def __init__(self):
        self.count = 0
    @classmethod
    def make_aggregator(cls) -> 'CountAggregator':
        return cls()
    def update(self, val: int) -> None:
        if val is not None:
            self.count += 1
    def value(self) -> int:
        return self.count

count = Function(
    IntType(), [IntType()],
    init_fn=CountAggregator.make_aggregator, update_fn=CountAggregator.update, value_fn=CountAggregator.value)


class MeanAggregator:
    def __init__(self):
        self.sum = 0
        self.count = 0
    @classmethod
    def make_aggregator(cls) -> 'MeanAggregator':
        return cls()
    def update(self, val: int) -> None:
        if val is not None:
            self.sum += val
            self.count += 1
    def value(self) -> float:
        if self.count == 0:
            return None
        return self.sum / self.count

mean = Function(
    FloatType(), [IntType()],
    init_fn=MeanAggregator.make_aggregator, update_fn=MeanAggregator.update, value_fn=MeanAggregator.value)


__all__ = [
    udf_call,
    cast,
    dict_map,
    sum,
    count,
    mean,
]

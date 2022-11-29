from typing import Callable, List
from pixeltable.type_system import ColumnType, StringType, IntType, DictType


class Function:
    def __init__(self, eval_fn: Callable, return_type: ColumnType, param_types: List[ColumnType]):
        self.eval_fn = eval_fn
        self.return_type = return_type
        self.param_types = param_types

    def __call__(self, *args) -> 'pixeltable.exprs.FunctionCall':
        self.check_args(args)
        from pixeltable import exprs
        return exprs.FunctionCall(self.eval_fn, self.return_type, args)

    def check_args(self, *args) -> None:
        """
        Verify that args match self.param_types.
        """
        pass


dict_map = Function(lambda s, d: d[s], IntType(), [StringType(), DictType()])

__all__ = [
    Function,
    dict_map
]


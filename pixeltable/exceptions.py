from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable import exprs


class Error(Exception):
    pass


@dataclass
class ExprEvalError(Exception):
    expr: 'exprs.Expr'
    expr_msg: str
    exc: Exception
    exc_tb: TracebackType
    input_vals: list[Any]
    row_num: int


class PixeltableWarning(Warning):
    pass

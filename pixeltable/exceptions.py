from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable import exprs


class Error(Exception):
    pass


class ExprEvalError(Exception):
    """
    Used during query execution to signal expr evaluation failures.

    NOT A USER-FACING EXCEPTION. All ExprEvalError instances need to be converted into Error instances.
    """

    expr: 'exprs.Expr'
    expr_msg: str
    exc: Exception
    exc_tb: TracebackType
    input_vals: list[Any]
    row_num: int

    def __init__(
        self,
        expr: 'exprs.Expr',
        expr_msg: str,
        exc: Exception,
        exc_tb: TracebackType,
        input_vals: list[Any],
        row_num: int,
    ) -> None:
        exct = type(exc)
        super().__init__(
            f'Expression evaluation failed with an error of type `{exct.__module__}.{exct.__qualname__}`:\n{expr}'
        )
        self.expr = expr
        self.expr_msg = expr_msg
        self.exc = exc
        self.exc_tb = exc_tb
        self.input_vals = input_vals
        self.row_num = row_num


class PixeltableWarning(Warning):
    pass

import enum
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pixeltable import exprs


class ErrorCategory(enum.Enum):
    """Category for user-facing errors. Used to classify each Error for handling/retries."""

    RETRYABLE = 'retryable'  # Conflict/concurrency/timeout; retry may succeed
    NOT_FOUND = 'not_found'  # Unknown columns, indexes, tables, functions, UDFs
    BAD_REQUEST = 'bad_request'  # Invalid input or operation not allowed (e.g. immutable)
    INTERNAL_SERVER_ERROR = 'internal_server_error'  # DB/config/runtime failure


# Short aliases for raise sites: excs.Error('...', excs.NOT_FOUND)
RETRYABLE = ErrorCategory.RETRYABLE
NOT_FOUND = ErrorCategory.NOT_FOUND
BAD_REQUEST = ErrorCategory.BAD_REQUEST
INTERNAL_SERVER_ERROR = ErrorCategory.INTERNAL_SERVER_ERROR


class Error(Exception):
    """Base exception for all user-facing Pixeltable errors."""

    category: ErrorCategory | None

    def __init__(self, msg: str, category: ErrorCategory | None = None) -> None:
        super().__init__(msg)
        self.category = category


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
